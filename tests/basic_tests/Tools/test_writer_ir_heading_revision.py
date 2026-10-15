'''Writer IR revision regression tests through the public artifact-based flow.'''
from unittest.mock import MagicMock, patch

import pytest

from lazyllm.tools.writer.data_models import WriterBlock, WriterDocument, WriterSpan, WritingContext, WritingTask
from lazyllm.tools.writer.data_models.revision import PatchResult, PatchSet
from lazyllm.tools.writer.tools.revision_tools import WriterRevisionTools
from lazyllm.tools.writer.utils import load_artifact_json, serialize_artifact_json


def _document(flat=False):
    source = WriterDocument(document_id='revision', title='规则', blocks=[
        WriterBlock(node_id='chapter', type='heading', numbering={'level': 1}, children=[
            WriterBlock(node_id='section', type='heading', content='售后', numbering={'level': 2}, children=[
                WriterBlock(node_id='body', type='paragraph', content='保留正文',
                            spans=[WriterSpan(text='保留正文', style={'bold': True})],
                            references=[{'type': 'citation', 'id': 'REF-1'}]),
                WriterBlock(node_id='image', type='image', content='流程图', editable=False,
                            provider_binding={'block_id': 'provider-image'}),
                WriterBlock(node_id='table', type='table', children=[
                    WriterBlock(node_id='row', type='table_row', children=[
                        WriterBlock(node_id='cell', type='table_cell', content='100'),
                    ]),
                ]),
                WriterBlock(node_id='subsection', type='heading', numbering={'level': 3}, children=[
                    WriterBlock(node_id='sub-body', type='paragraph', content='子节正文'),
                ]),
            ]),
            WriterBlock(node_id='later-section', type='heading', numbering={'level': 2}),
        ]),
        WriterBlock(node_id='next', type='heading', numbering={'level': 1}),
    ])
    if flat:
        # Feishu headings and body blocks are siblings; table cells stay nested.
        source.blocks = [block for block in source.iter_blocks() if block.type not in {'table_row', 'table_cell'}]
        for block in source.blocks:
            if block.type == 'heading':
                block.children = []
    return source


def _edit(operation, node_id, **fields):
    return {'modify_type': operation, 'content_ref': {'node_id': node_id},
            'instruction': '按指定范围修改，保留其他内容。', **fields}


def _revise(tmp_path, source, instructions, generated=None, *, title_only=False):
    source_path = tmp_path / 'source_ir.lmd'
    original = serialize_artifact_json(source)
    source_path.write_text(original, encoding='utf-8')
    document = str(source_path)
    task = WritingTask(task_id='revision', task_type='revise', query='按要求修订文档。')
    context = WritingContext(context_id='revision')
    targets = [{'content_ref': item['content_ref']} for item in instructions]
    responses = [{'target_title': title_only, 'targets': targets}]
    if not title_only:
        responses.append({'scope': 'block', 'instructions': [
            {'instruction_id': f'edit-{i}', **item} for i, item in enumerate(instructions, 1)
        ]})
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
    result = load_artifact_json(applied['artifact_path'], PatchResult)
    assert result.success and not result.failed_hunks
    assert source_path.read_text(encoding='utf-8') == original
    revised = load_artifact_json(applied['metadata']['artifact_paths']['revised_document'], WriterDocument)
    return revised, load_artifact_json(changes['artifact_path'], PatchSet)


def test_heading_update_flow_preserves_document_title_and_subtree(tmp_path):
    source = _document()
    revised, changes = _revise(tmp_path, source, [_edit('update', 'section')], {'changes': {'edit-1': [{
        'type': 'heading', 'content': '售后服务', 'children': [{'type': 'paragraph', 'content': '模型附带正文'}],
    }]}})
    expected = source.model_copy(deep=True)
    expected.block_by_id('section').content = '售后服务'
    assert revised == expected
    assert [(item.modify_type, item.target_node_id) for item in changes.hunks] == [('update', 'section')]


def test_document_title_flow_preserves_all_blocks(tmp_path):
    source = _document()
    revised, changes = _revise(tmp_path, source, [], {'new_title': '服务规则'}, title_only=True)
    assert revised.title == '服务规则'
    assert revised.blocks == source.blocks
    assert changes.hunks == []


def test_update_flow_preserves_format_references_and_other_blocks(tmp_path):
    source = _document()
    spans = [WriterSpan(text='更新正文', style={'bold': True})]
    revised, _ = _revise(tmp_path, source, [_edit('update', 'body')], {'changes': {
        'edit-1': [{'spans': [span.model_dump() for span in spans]}],
    }})
    expected = source.model_copy(deep=True)
    expected.block_by_id('body').content = '更新正文'
    expected.block_by_id('body').spans = spans
    assert revised == expected


def test_delete_section_flow_removes_owned_subtree_only(tmp_path):
    source = _document()
    revised, _ = _revise(tmp_path, source, [_edit('delete', 'section')])
    expected = source.model_copy(deep=True)
    expected.blocks[0].children.pop(0)
    assert revised == expected


def test_move_flow_preserves_original_block_and_other_content(tmp_path):
    source = _document()
    revised, _ = _revise(tmp_path, source, [_edit(
        'move', 'body', position='after', destination_ref={'node_id': 'later-section'},
    )])
    expected = source.model_copy(deep=True)
    body = expected.block_by_id('section').children.pop(0)
    expected.blocks[0].children.append(body)
    assert revised == expected


@pytest.mark.parametrize('flat', [False, True], ids=['nested', 'flat'])
def test_create_flow_distinguishes_heading_and_section_without_reparenting(tmp_path, flat):
    source = _document(flat)
    revised, changes = _revise(tmp_path, source, [
        _edit('create', 'section', position='after', anchor_scope='block'),
        _edit('create', 'section', position='after', anchor_scope='section'),
    ], {'changes': {
        'edit-1': [{'type': 'paragraph', 'content': '标题下的说明'}],
        'edit-2': [{'type': 'heading', 'content': '新增节', 'numbering': {'level': 2},
                    'children': [{'type': 'paragraph', 'content': '新增正文'}]}],
    }})
    assert [item.modify_type for item in changes.hunks] == ['create', 'create']
    by_content = {block.content: block for block in revised.iter_blocks()}
    note, section = by_content['标题下的说明'], by_content['新增节']
    assert section.numbering == {'level': 2}
    assert [child.content for child in section.children] == ['新增正文']
    expected = source.model_copy(deep=True)
    if flat:
        expected.blocks.insert(2, note)
        expected.blocks.insert(next(i for i, block in enumerate(expected.blocks) if block.node_id == 'later-section'),
                               section)
    else:
        expected.block_by_id('section').children.insert(0, note)
        expected.blocks[0].children.insert(1, section)
    assert revised == expected
