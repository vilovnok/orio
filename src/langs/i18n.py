import gettext
import os
from typing import Dict

from src.utils.config import inited_config as config


class I18nRuntimeException(Exception):
    pass


class I18n:
    LANG_RU = 'ru'
    LANG_CODE_RU = 'ru_RU'
    LANG_EN = 'en'
    LANG_CODE_EN = 'en_US'
    LANG_ES = 'es'
    LANG_CODE_ES = 'es_ES'
    LANG_PT = 'pt'
    LANG_CODE_PT = 'pt_PT'
    LANG_ID = 'id'
    LANG_CODE_ID = 'id_ID'
    LANG_TR = 'tr'
    LANG_CODE_TR = 'tr_TR'

    AVAILABLE_LANGS: Dict[str, Dict[str, str]] = {
        LANG_RU: {'locale': LANG_CODE_RU},
        LANG_EN: {'locale': LANG_CODE_EN},
        LANG_ES: {'locale': LANG_CODE_ES},
        LANG_PT: {'locale': LANG_CODE_PT},
        LANG_ID: {'locale': LANG_CODE_ID},
        LANG_TR: {'locale': LANG_CODE_TR},
    }

    def __init__(self):
        self.locales_dir = "/Users/richardgurtsiev/Desktop/projects/multiagent_system/src/langs/locales"
        self.curr_domain = self._init_domain()
        self.translators = {}
        self._add_translations()

    def _init_domain(self) -> str:
        version_file = os.path.join(self.locales_dir, 'version.txt')
        if not os.path.exists(version_file):
            raise I18nRuntimeException('Locales version not found')

        with open(version_file, 'r') as f:
            version = f.read().strip()

        if not version.isdigit() or int(version) <= 0:
            raise I18nRuntimeException('Invalid locales version')

        return f'messages_{version}'

    def _add_translations(self):
        for lang, params in self.AVAILABLE_LANGS.items():
            locale = params['locale']
            mo_path = os.path.join(
                self.locales_dir,
                locale,
                'LC_MESSAGES',
                f'{self.curr_domain}.mo'
            )
            if os.path.exists(mo_path):
                with open(mo_path, 'rb') as f:
                    self.translators[lang] = gettext.GNUTranslations(f)
            else:
                self.translators[lang] = gettext.NullTranslations()

    def get_translator(self, lang: str) -> gettext.NullTranslations:
        if lang not in self.translators:
            lang = self.LANG_EN
        return self.translators[lang]

    def is_lang_available(self, lang: str) -> bool:
        return lang in self.AVAILABLE_LANGS


i18n = I18n()
