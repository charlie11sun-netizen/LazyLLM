'''Markdown revision regression tests; only model responses are mocked.'''
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lazyllm.tools.writer.data_models import WritingContext, WritingTask
from lazyllm.tools.writer.data_models.revision import StringReplaceResult
from lazyllm.tools.writer.tools.revision_tools import WriterRevisionTools
from lazyllm.tools.writer.utils import load_artifact_json


PARAGRAPH = '线上办理。付款后保存凭证[^ref]。'
ANCHOR = '付款后保存凭证[^ref]。'
IMAGE = '![办理流程](plots/run_(1).png)'
SECTION = ('# 办理\n\n' + PARAGRAPH + '\n\n线下办理。\n\n退款旧条款。\n\n' + IMAGE
           + '\n\n|项目|值|\n|---|---|\n|费用|100|\n\n## 补充说明\n\n子节正文。\n\n')
TAIL = '# 其他\n\n其他正文。\n\n[^ref]: 引用来源\n'
SOURCE = SECTION + TAIL


def _edit(operation, scope='paragraph', **fields):
    return {'modify_type': operation, 'target_scope': scope, 'content_ref': {'heading_path': ['办理']},
            'instruction': '按指定范围修改，保留其他内容。', **fields}


def _revise(tmp_path, instructions, generated=None):
    source_path = tmp_path / 'source.md'
    source_path.write_text(SOURCE, encoding='utf-8')
    document = str(source_path)
    task = WritingTask(task_id='revision', task_type='revise', query='按要求修订办理章节。')
    context = WritingContext(context_id='revision')
    responses = [
        {'target_title': False, 'targets': [{'content_ref': {'heading_path': ['办理']}}]},
        {'scope': 'block', 'instructions': [
            {'instruction_id': f'edit-{i}', **item} for i, item in enumerate(instructions, 1)
        ]},
    ]
    if generated is not None:
        responses.append(generated)
    model = MagicMock(side_effect=responses)
    tool = WriterRevisionTools(llm=MagicMock(), artifact_store=str(tmp_path))
    with patch.object(tool, '_build_structured_llm', return_value=model):
        located = tool.locate_revision_target(task, document, context)
        planned = tool.generate_modify_plan(task, document, located['artifact_path'], context)
        changes = tool.generate_revision_set(document, planned['artifact_path'], context)
        applied = tool.apply_revision(document, changes['artifact_path'], context)
    assert model.call_count == len(responses)
    result = load_artifact_json(applied['artifact_path'], StringReplaceResult)
    assert result.success and result.applied_replacements
    assert source_path.read_text(encoding='utf-8') == SOURCE
    return Path(applied['metadata']['artifact_paths']['revised_document']).read_text(encoding='utf-8')


@pytest.mark.parametrize('scope,position,expected', [
    ('paragraph', 'before', '新增说明。\n\n' + PARAGRAPH),
    ('paragraph', 'after', PARAGRAPH + '\n\n新增说明。'),
    ('fragment', 'before', '线上办理。新增说明。' + ANCHOR),
    ('fragment', 'after', PARAGRAPH + '新增说明。'),
])
def test_create_flow_respects_scope_and_position(tmp_path, scope, position, expected):
    revised = _revise(tmp_path, [_edit('create', scope, position=position, locator_text=ANCHOR)],
                      {'insert_text': '新增说明。'})
    assert revised == SOURCE.replace(PARAGRAPH, expected)


def test_batch_flow_keeps_separate_anchors_and_replaces_only_requested_paragraph(tmp_path):
    revised = _revise(tmp_path, [
        _edit('create', position='after', locator_text=PARAGRAPH),
        _edit('create', position='after', locator_text='线下办理。'),
        _edit('update', locator_text='退款旧条款。', instruction='整段替换为“退款新条款。”。'),
    ], {
        'insertions': {'instruction-1': {'insert_text': '线上咨询。'},
                       'instruction-2': {'insert_text': '线下咨询。'}},
        'contents': {'instruction-3': {'new_string': '退款新条款。'}},
    })
    expected = SOURCE.replace(PARAGRAPH, PARAGRAPH + '\n\n线上咨询。')
    expected = expected.replace('线下办理。', '线下办理。\n\n线下咨询。').replace('退款旧条款。', '退款新条款。')
    assert revised == expected


@pytest.mark.parametrize('scope,locator,removed', [
    ('paragraph', '退款旧条款。', '退款旧条款。'),
    ('image', '办理流程', IMAGE),
    ('section', None, SECTION),
])
def test_delete_flow_uses_target_scope_and_preserves_remaining_document(tmp_path, scope, locator, removed):
    revised = _revise(tmp_path, [_edit('delete', scope, locator_text=locator)])
    assert revised == SOURCE.replace(removed, '')


def test_move_flow_preserves_paragraph_and_other_content(tmp_path):
    revised = _revise(tmp_path, [_edit(
        'move', locator_text='线下办理。', position='after', destination_ref={'heading_path': ['其他']},
        destination_scope='paragraph', destination_locator_text='其他正文。',
    )])
    expected = SOURCE.replace('线下办理。', '').replace('其他正文。', '其他正文。\n\n线下办理。')
    assert revised == expected


def test_create_after_section_flow_keeps_subsections_in_place(tmp_path):
    revised = _revise(tmp_path, [_edit('create', 'section', position='after')], {'insert_text': '新增说明。'})
    assert revised == SECTION + '新增说明。\n\n' + TAIL


def test_flow_stops_when_plan_changes_content_outside_located_scope(tmp_path):
    with pytest.raises(ValueError, match='not present in locate_result.targets'):
        _revise(tmp_path, [_edit('update', locator_text='其他正文。', content_ref={'heading_path': ['其他']})])
    assert not list(tmp_path.rglob('revised_document*'))
    assert (tmp_path / 'source.md').read_text(encoding='utf-8') == SOURCE
