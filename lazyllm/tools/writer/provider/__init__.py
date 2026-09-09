from .base import (
    WriterProviderBase,
    WriterProviderCapabilities,
    WriterProviderCapability,
    WriterProviderCapabilityError,
    WriterProviderDocument,
    WriterProviderRevisionError,
    WriterProviderWriteMode,
    WriterProviderWriteOutcomeError,
    is_ambiguous_write_error,
)
from .feishu import FeishuWriterProvider
from .github import GitHubWriterProvider
from .notion import NotionWriterProvider
from .wechat import WeChatWriterProvider
from .registry import (
    get_writer_provider,
    match_writer_provider,
    register_writer_provider,
    resolve_writer_create_target,
)


register_writer_provider(FeishuWriterProvider)
register_writer_provider(GitHubWriterProvider)
register_writer_provider(NotionWriterProvider)
register_writer_provider(WeChatWriterProvider)


__all__ = [
    'FeishuWriterProvider',
    'GitHubWriterProvider',
    'NotionWriterProvider',
    'WeChatWriterProvider',
    'WriterProviderBase',
    'WriterProviderCapabilities',
    'WriterProviderCapability',
    'WriterProviderCapabilityError',
    'WriterProviderDocument',
    'WriterProviderRevisionError',
    'WriterProviderWriteMode',
    'WriterProviderWriteOutcomeError',
    'is_ambiguous_write_error',
    'get_writer_provider',
    'match_writer_provider',
    'register_writer_provider',
    'resolve_writer_create_target',
]
