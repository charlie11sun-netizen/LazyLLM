from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lazyllm.tools.writer.data_models import ContentRef, WritingContext, WritingTask
from lazyllm.tools.writer.data_models.revision import MarkdownModifyPlan, StringReplaceSet
from lazyllm.tools.writer.tools.revision_tools import WriterRevisionTools
from lazyllm.tools.writer.utils import load_artifact_json


PRESERVED = '\n\n![保留图片](keep.png)\n\n|项目|值|\n|---|---|\n|保留|1|\n\n[^ref]: 保留引用\n'


@pytest.fixture
def tool(tmp_path):
    return WriterRevisionTools(llm=MagicMock(), artifact_store=str(tmp_path))


def _plan(**fields):
    return MarkdownModifyPlan(scope='block', instructions=[{
        'instruction_id': 'edit-1', 'modify_type': 'create', 'position': 'after',
        'instruction': '按指定范围修改，保留其他内容。',
        'content_ref': {'heading_path': ['A']}, 'target_scope': 'paragraph', **fields,
    }])


def _apply(tool, source, plan, response=None):
    model = MagicMock(return_value=response) if response is not None else MagicMock(
        side_effect=AssertionError('Deletion should use program location without a model call'))
    context = WritingContext(context_id='scoped-revision')
    with patch.object(tool, '_build_structured_llm', return_value=model):
        generated = tool.generate_string_replace_set(source, plan, context)
        replacements = load_artifact_json(generated['artifact_path'], StringReplaceSet)
        applied = tool.apply_string_replace(source, replacements, context)
    assert model.call_count == int(response is not None)
    return Path(applied['revised_document_md']).read_bytes().decode()


def _normalize(tool, plan):
    reference = ContentRef(heading_path=['A'])
    return tool._normalize_modify_plan(
        plan, WritingTask(task_type='revise', query='在指定位置插入内容。'),
        [reference], {tool._content_ref_key(reference)},
    )


@pytest.mark.parametrize('scope,position,expected', [
    ('paragraph', 'before', '新增句。\n\n第一句。定位句。末句[^ref]。'),
    ('paragraph', 'after', '第一句。定位句。末句[^ref]。\n\n新增句。'),
    ('fragment', 'before', '第一句。新增句。定位句。末句[^ref]。'),
    ('fragment', 'after', '第一句。定位句。新增句。末句[^ref]。'),
])
def test_insert_respects_paragraph_or_fragment_boundary(tool, scope, position, expected):
    paragraph = '第一句。定位句。末句[^ref]。'
    source = '# A\n\n' + paragraph + PRESERVED
    revised = _apply(tool, source, _plan(
        target_scope=scope, position=position, locator_text='定位句。',
    ), {'insert_text': '新增句。'})
    assert revised == source.replace(paragraph, expected)


def test_batch_keeps_inserted_text_separate_from_complete_update(tool):
    source = '# A\n\n第一句。保留句。\n\n旧前缀，待替换段落。' + PRESERVED
    plan = _plan(target_scope='fragment', locator_text='第一句。')
    plan.instructions.append(plan.instructions[0].model_copy(update={
        'instruction_id': 'edit-2', 'modify_type': 'update', 'target_scope': 'paragraph',
        'locator_text': '待替换段落', 'instruction': '整段替换为“指定新段落。”。',
    }))
    revised = _apply(tool, source, plan, {
        'insertions': {'instruction-1': {'insert_text': '新增句。'}},
        'contents': {'instruction-2': {'new_string': '指定新段落。'}},
    })
    assert revised == '# A\n\n第一句。新增句。保留句。\n\n指定新段落。' + PRESERVED


def test_two_insertion_anchors_remain_separate(tool):
    source = '# A\n\n线上办理。\n\n线下办理。' + PRESERVED
    plan = _plan(locator_text='线上办理。')
    plan.instructions.append(plan.instructions[0].model_copy(update={
        'instruction_id': 'edit-2', 'locator_text': '线下办理。',
    }))
    revised = _apply(tool, source, _normalize(tool, plan), {'insertions': {
        'instruction-1': {'insert_text': '线上咨询。'},
        'instruction-2': {'insert_text': '线下咨询。'},
    }})
    assert revised == '# A\n\n线上办理。\n\n线上咨询。\n\n线下办理。\n\n线下咨询。' + PRESERVED


@pytest.mark.parametrize('scope,locator,second_scope,count', [
    ('paragraph', '定位句。', 'fragment', 2),
    ('paragraph', None, 'paragraph', 2),
    ('fragment', None, 'fragment', 2),
    ('paragraph', '定位句。', 'paragraph', 1),
])
def test_merge_requires_same_scope_and_known_anchor(tool, scope, locator, second_scope, count):
    plan = _plan(target_scope=scope, locator_text=locator)
    plan.instructions.append(plan.instructions[0].model_copy(update={
        'instruction_id': 'edit-2', 'target_scope': second_scope,
    }))
    assert len(_normalize(tool, plan).instructions) == count


@pytest.mark.parametrize('target,locator', [
    ('![目标说明](target.png)', '目标说明'),
    ('![目标说明](plots/run_(1).png)', '![目标说明]'),
    (r'![目标说明](plots/run_\(1\).png)', '目标说明'),
    ('![目标说明](<plots/run 1).png>)', '目标说明'),
    ('![目标说明](target.png "title ) text")', '目标说明'),
    ('<img src="target.png" alt="目标说明">', '目标说明'),
])
def test_image_delete_resolves_complete_source_in_target_section(tool, target, locator):
    source = '# A\n\n正文提到目标说明[^ref]。\n\n' + target + PRESERVED + '\n# B\n\n![目标说明](other.png)\n'
    revised = _apply(tool, source, _plan(modify_type='delete', target_scope='image', locator_text=locator))
    assert revised == source.replace(target, '')


def test_delete_image_and_text_preserves_neighboring_content(tool):
    target = '![目标说明](target.png)'
    source = '# A\n\n保留前句。删除这句话。保留后句。\n\n' + target + PRESERVED
    plan = _plan(modify_type='delete', target_scope='image', locator_text='目标说明')
    plan.instructions.append(plan.instructions[0].model_copy(update={
        'instruction_id': 'edit-2', 'target_scope': 'fragment', 'locator_text': '删除这句话。',
    }))
    assert _apply(tool, source, plan) == source.replace(target, '').replace('删除这句话。', '')


@pytest.mark.parametrize('operation', ['create', 'delete'])
def test_section_boundary_includes_descendants(tool, operation):
    before = '# 保留前节\n\n前文。\n\n'
    section = '# A\n\n正文。\n\n## 子节\n\n子节正文。' + PRESERVED + '\n'
    after = '# 保留后节\n\n后文。\n'
    response = {'insert_text': '新增正文。'} if operation == 'create' else None
    revised = _apply(tool, before + section + after, _plan(
        modify_type=operation, target_scope='section',
    ), response)
    expected = before + (section + '新增正文。\n\n' if operation == 'create' else '') + after
    assert revised == expected
