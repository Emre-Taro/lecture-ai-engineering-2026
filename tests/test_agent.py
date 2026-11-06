"""エージェントの配線テスト。

TestModel は実際のLLMを呼ばない偽モデルで、登録されたツールを全部呼んでから
ダミーの回答を返す。APIキーも利用枠も使わずに、
「ツールが登録され、呼ばれ、結果がモデルに戻るか」を確かめられる。
"""

from pydantic_ai import DeferredToolRequests, DeferredToolResults
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel

from app.agent import agent, print_trace

BASE_TOOLS = {"search_glossary", "get_lecture_summary", "make_quiz", "save_learning_log"}
NON_APPROVAL_TOOLS = BASE_TOOLS - {"save_learning_log"}


def _parts(result, kind):
    return [p for m in result.all_messages() for p in m.parts if isinstance(p, kind)]


def test_base_tools_are_registered_and_called():
    model = TestModel(call_tools=sorted(NON_APPROVAL_TOOLS))
    with agent.override(model=model):
        result = agent.run_sync("LoRAとは？")

    offered = {t.name for t in model.last_model_request_parameters.function_tools}
    assert BASE_TOOLS <= offered  # 演習でツールを足しても壊れないよう「含む」で判定

    called = {p.tool_name for p in _parts(result, ToolCallPart)}
    assert NON_APPROVAL_TOOLS <= called
    assert len(_parts(result, ToolReturnPart)) == len(_parts(result, ToolCallPart))
    assert result.output


def test_glossary_result_reaches_the_model():
    model = TestModel(call_tools=["search_glossary"])
    with agent.override(model=model):
        result = agent.run_sync("RAGとは？")
    returns = _parts(result, ToolReturnPart)
    assert returns and returns[0].tool_name == "search_glossary"


def test_print_trace_shows_full_tool_return_and_call_count(capsys):
    model = TestModel(call_tools=["search_glossary"])
    with agent.override(model=model):
        result = agent.run_sync("RAGとは？", model=model)

    returned_content = str(_parts(result, ToolReturnPart)[0].content)
    print_trace(result)

    output = capsys.readouterr().out
    assert returned_content in output
    assert "[tool call 1] search_glossary" in output
    assert "[tool calls] 1" in output


def test_save_learning_log_waits_for_approval(tmp_path, monkeypatch):
    from app import tools

    monkeypatch.setattr(tools, "LOG_DIR", tmp_path / "learning_log")
    model = TestModel(call_tools=["save_learning_log"])
    with agent.override(model=model):
        pending = agent.run_sync("この内容を学習記録に保存して", model=model)

    assert isinstance(pending.output, DeferredToolRequests)
    assert not (tmp_path / "learning_log").exists()

    call_id = pending.output.approvals[0].tool_call_id
    continuation_model = TestModel(call_tools=[])
    with agent.override(model=continuation_model):
        result = agent.run_sync(
            message_history=pending.all_messages(),
            deferred_tool_results=DeferredToolResults(approvals={call_id: True}),
            model=continuation_model,
        )

    saved = (tmp_path / "learning_log" / "learning_log.md").read_text(encoding="utf-8")
    assert saved.strip()
    assert result.output
