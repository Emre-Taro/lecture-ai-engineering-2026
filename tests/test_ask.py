from types import SimpleNamespace

from app.ask import ask, main


def test_ask_prints_input_and_output_token_counts(monkeypatch, capsys):
    response = SimpleNamespace(
        text="回答",
        usage_metadata=SimpleNamespace(
            prompt_token_count=12,
            candidates_token_count=8,
        ),
    )

    class FakeChat:
        def send_message(self, question):
            assert question == "質問"
            return response

    assert ask("質問", FakeChat()).text == "回答"
    assert capsys.readouterr().out.strip() == "[tokens] input: 12, output: 8"


def test_main_reuses_chat_for_follow_up_questions(monkeypatch, capsys):
    class FakeChat:
        def __init__(self):
            self.questions = []

        def send_message(self, question):
            self.questions.append(question)
            return SimpleNamespace(text=f"回答: {question}", usage_metadata=None)

    chat = FakeChat()

    class FakeClient:
        class chats:
            @staticmethod
            def create(**kwargs):
                return chat

    monkeypatch.setattr("app.ask.genai.Client", lambda **kwargs: FakeClient())
    monkeypatch.setattr("sys.argv", ["app.ask", "最初の質問"])
    answers = iter(["", "続けて質問", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))

    main()

    assert chat.questions == ["最初の質問", "続けて質問"]
    assert "回答: 最初の質問" in capsys.readouterr().out


def test_compact_history_summarizes_old_turns_and_keeps_recent_turns():
    from app.ask import compact_history

    def message(role, text):
        return SimpleNamespace(role=role, parts=[SimpleNamespace(text=text)])

    old_history = [
        message("user", "古い質問"),
        message("model", "古い回答"),
        message("user", "最近の質問1"),
        message("model", "最近の回答1"),
        message("user", "最近の質問2"),
        message("model", "最近の回答2"),
    ]

    class FakeChat:
        def get_history(self, curated=True):
            return old_history

    class FakeClient:
        class models:
            @staticmethod
            def generate_content(**kwargs):
                assert "古い質問" in kwargs["contents"]
                return SimpleNamespace(text="要約済みの古い会話")

        class chats:
            @staticmethod
            def create(**kwargs):
                return SimpleNamespace(**kwargs)

    compacted_chat, summary = compact_history(FakeClient(), FakeChat(), None)

    assert summary == "要約済みの古い会話"
    assert compacted_chat.history == old_history[-4:]
    assert "要約済みの古い会話" in compacted_chat.config.system_instruction