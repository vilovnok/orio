from src.app.schemes.agent import CopilotAnswersScheme
from src.langs.i18n import i18n
from typing import Optional


class PlugsBuilder:
    def __init__(self, language: Optional[str] = None) -> None:
        if language is None:
            language = i18n.LANG_EN
        self.translator = i18n.get_translator(language)

    def unable_to_use(self) -> CopilotAnswersScheme:
        """Return plug for unable to use feature.

        Returns:
            CopilotAnswersScheme(is_plug=True)"""
        return CopilotAnswersScheme(
            text=self.translator.gettext(
                'Unable to use this feature. '
                'Please contact our support team or try again later.'
            ),
            is_plug=True,
            action='unknown'
        )

    def info_not_found(self) -> CopilotAnswersScheme:
        """Return plug for could not find answer if kb.

        Returns:
            CopilotAnswersScheme(is_support=True)"""
        return CopilotAnswersScheme(
            text=self.translator.gettext(
                "I couldn't find an answer to that in db's "
                "knowledge base or online. "
                "If you need help, "
                "feel free to reach out to our support team."
            ),
            is_support=True,
            action='unknown'
        )
