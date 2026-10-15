from unittest.mock import MagicMock, patch

import pytest

from lazyllm.tools.writer.data_models import (
    ContentRef, LocateResult, ModifyInstruction, ModifyPlan, PatchHunk, PatchSet,
    WriterBlock, WriterDocument, WriterSpan, WritingContext, WritingTask,
)
from lazyllm.tools.writer.data_models.revision import GeneratedRevision, RevisionBlockContent
from lazyllm.tools.writer.tools.revision_tools import WriterRevisionTools, apply_patch_to_ir
from lazyllm.tools.writer.utils import load_artifact_json


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
        # Feishu headings and their body blocks are siblings; table cells stay nested.
        source.blocks = [block for block in source.iter_blocks() if block.type not in {'table_row', 'table_cell'}]
        for block in source.blocks:
            if block.type == 'heading':
                block.children = []
    return source


def _apply(source, instruction, content):
    plan = ModifyPlan(scope='block', instructions=[ModifyInstruction(instruction_id='edit', **instruction)])
    generated = GeneratedRevision(changes={'edit': [content]})
    patch_set = WriterRevisionTools()._compile_generated_revision(source, plan, generated)
    revised, result = apply_patch_to_ir(source, PatchSet.model_validate_json(patch_set.model_dump_json()))
    assert result.success
    return revised, patch_set


def test_heading_rename_preserves_subtree_and_document_title():
    source = _document()
    revised, patch_set = _apply(source, {
        'modify_type': 'update', 'content_ref': ContentRef(node_id='section'), 'instruction': '只修改章节标题。',
    }, RevisionBlockContent(type='heading', content='售后服务', children=[
        RevisionBlockContent(type='paragraph', content='模型附带的错误子树'),
    ]))
    expected = source.model_copy(deep=True)
    expected.block_by_id('section').content = '售后服务'
    assert revised == expected
    assert len(patch_set.hunks) == 1
    assert source.block_by_id('section').content == '售后'


def test_document_title_only_skips_block_planning(tmp_path):
    source = _document()
    tool = WriterRevisionTools(llm=MagicMock(), artifact_store=str(tmp_path))
    context = WritingContext(context_id='title')
    with patch.object(tool, '_call_llm_structured', return_value=GeneratedRevision(new_title='服务规则')) as model:
        planned = tool.generate_modify_plan(
            WritingTask(task_type='revise', query='把文档总标题改为服务规则。'), source,
            LocateResult(doc_id=source.document_id, target_title=True), context,
        )
        model.assert_not_called()
        output = tool.generate_patch_set(source, planned['artifact_path'], context)
    revised, result = apply_patch_to_ir(source, load_artifact_json(output['artifact_path'], PatchSet))
    assert result.success
    assert revised.title == '服务规则'
    assert revised.blocks == source.blocks


def test_candidates_include_parent_and_heading_level():
    candidates = WriterRevisionTools()._revision_candidates(_document())
    actual = [(item['content_ref'].node_id, item['parent_node_id'], item['heading_level']) for item in candidates]
    assert actual == [
        ('chapter', None, 1), ('section', 'chapter', 2), ('body', 'section', None),
        ('image', 'section', None), ('table', 'section', None), ('row', 'table', None),
        ('cell', 'row', None), ('subsection', 'section', 3), ('sub-body', 'subsection', None),
        ('later-section', 'chapter', 2), ('next', None, 1),
    ]


@pytest.mark.parametrize('flat', [False, True], ids=['nested', 'flat'])
@pytest.mark.parametrize('anchor,scope,position,next_id,parent', [
    ('section', 'section', 'before', 'section', 'chapter'),
    ('section', 'section', 'after', 'later-section', 'chapter'),
    ('chapter', 'section', 'after', 'next', None),
    ('next', 'section', 'after', None, None),
    ('section', 'block', 'after', 'body', 'section'),
    ('image', 'block', 'after', 'table', 'section'),
])
def test_create_uses_requested_boundary_without_moving_existing_nodes(flat, anchor, scope, position, next_id, parent):
    source = _document(flat)
    original = source.model_copy(deep=True)
    revised, patch_set = _apply(source, {
        'modify_type': 'create', 'content_ref': ContentRef(node_id=anchor), 'instruction': '在指定位置新增章节。',
        'position': position, 'anchor_scope': scope,
    }, RevisionBlockContent(type='heading', content='新增节', numbering={'level': 2}, children=[
        RevisionBlockContent(type='paragraph', content='新增正文'),
    ]))
    hunk, = patch_set.hunks
    assert hunk.modify_type == 'create'
    assert hunk.parent_node_id == (None if flat else parent)
    order = [block.node_id for block in revised.iter_blocks()]
    inserted_index = order.index(hunk.target_node_id)
    assert order[inserted_index + 2:inserted_index + 3] == ([next_id] if next_id else [])
    assert revised.block_by_id(hunk.target_node_id).numbering == {'level': 2}
    # Removing only the addition must restore every old parent, block, style and reference.
    restored = revised.model_copy(deep=True)
    for blocks in [restored.blocks, *[block.children for block in restored.iter_blocks()]]:
        blocks[:] = [block for block in blocks if block.node_id != hunk.target_node_id]
    assert restored == original == source


@pytest.mark.parametrize('second_scope,count', [('block', 2), ('section', 1), (None, 2)])
def test_create_merge_respects_anchor_scope(second_scope, count):
    reference = ContentRef(node_id='section')
    plan = ModifyPlan(scope='block', instructions=[ModifyInstruction(
        modify_type='create', content_ref=reference, position='after', anchor_scope=scope, instruction='新增内容。',
    ) for scope in ['section', second_scope]])
    tool = WriterRevisionTools()
    normalized = tool._normalize_modify_plan(
        plan, WritingTask(task_type='revise', query='在指定位置新增内容。'),
        [reference], {tool._content_ref_key(reference)}, document=_document(),
    )
    assert len(normalized.instructions) == count


@pytest.mark.parametrize('heading_index,allowed', [(1, True), (0, False)])
def test_image_reparenting_requires_unchanged_reading_order(heading_index, allowed):
    source = WriterDocument(document_id='images', blocks=[
        WriterBlock(node_id='body', type='paragraph', content='保留正文'),
        WriterBlock(node_id='image', type='image', content='流程图', editable=False),
    ])
    patch_set = PatchSet(target_doc_id=source.document_id, hunks=[
        PatchHunk(modify_type='create', target_node_id='heading', index=heading_index,
                  block=WriterBlock(node_id='heading', type='heading', numbering={'level': 1})),
        PatchHunk(modify_type='move', target_node_id='image', parent_node_id='heading', index=0),
    ])
    if not allowed:
        with pytest.raises(ValueError, match='existing image blocks cannot be updated or moved'):
            apply_patch_to_ir(source, patch_set)
    else:
        revised, result = apply_patch_to_ir(source, patch_set)
        assert result.success
        assert revised.blocks[0] == source.blocks[0]
        assert revised.block_by_id('heading').children == [source.blocks[1]]
