import logging
from typing import Dict, Tuple, Optional, Any, cast
import pandas as pd
from pathlib import Path
from pydantic import BaseModel

from src.copilot.utils.contexts.service_definer import ClosestService

logger = logging.getLogger(__name__)


class TariffOpsScheme(BaseModel):
    entity: str
    description: str
    actual_plan: str
    available_map: Dict[str, bool]
    quota: Optional[int]

    def __init__(self, **data):
        # Нормализуем actual_plan при инициализации
        if 'actual_plan' in data:
            data['actual_plan'] = data['actual_plan'].strip()
        super().__init__(**data)

    def __repr__(self):
        return f"""\
        TariffOpsScheme(
            entity={self.entity},
            description={self.description},
            actual_plan={self.actual_plan},
            available_map={self.available_map},
            quota={self.quota}
        )"""


class TariffDefiner:
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.df = self._load_data()
        self.entity_map = self._build_entity_map()
        self.plans = {p.strip().lower(): p for p in self.df.index}
        # TODO: Костыль! Пример использования вместе с .csv файлом!
        self.trial_variants = {
            'пробный период',
            'trial',
            'período de prueba',
            'percobaan',
            'deneme'
        }
        self.plan_map = {
            'base': self.plans['base'],
            'advanced': self.plans['advanced'],
            'enterprise': self.plans['enterprise'],
            'partner': self.plans['enterprise'],
            'trial': self.plans['enterprise'],
            'shopify': self.plans['advanced'],
        }
        logger.info('plan_map: %r', self.plan_map)

    def _load_data(self) -> pd.DataFrame:
        """Read data from CSV file and set index to first column."""
        df = pd.read_csv(self.csv_path, header=[0, 1], sep=';')
        plan_col = df.columns[0]
        df = df.set_index(plan_col).rename_axis("plan")
        return df

    def _build_entity_map(self) -> Dict[str, Tuple[str, str]]:
        """Build entity map from DataFrame columns.
        Returns:
            Dict[str, Tuple[str, str]]: Entity map.
            Key: entity name in lowercase.
            Value: tuple of description and entity name."""

        entity_map: Dict[str, Tuple[str, str]] = {}

        # Типизация для MyPy (columns - это MultiIndex)
        cols = cast(pd.MultiIndex, self.df.columns)

        for desc, ent in cols:
            key = ent.strip().lower()
            entity_map[key] = (desc, ent)

        return entity_map

    def _parse_cell(self, val: Any) -> Dict[str, Any]:
        """Parse cell value to dict with available and quota.
        Returns:
            Dict[str, Any]: Dict with available and quota.
            Key: "available" or "quota".
            Value: bool or int.

        Example Output:
            >>> {'available': True, 'quota': 100}
            >>> {'available': False, 'quota': None}
            >>> {'available': True, 'quota': None}
            ...
        """

        if pd.isna(val) or str(val).strip() in {"", "-"}:
            return {"available": False, "quota": None}

        s = str(val).strip()
        if s == '+':
            return {"available": True, "quota": None}

        if s == 'unlimited':
            return {"available": True, "quota": None}

        if s.isdigit():
            return {"available": True, "quota": int(s)}

        return {"available": True, "quota": None}

    def get_entity_scheme(
        self,
        plan: str,
        service: ClosestService | str
    ) -> TariffOpsScheme:
        """Get entity scheme for given plan and entity.
        Args:
            plan: Plan name.
            service: Service name.
        Returns:
            TariffOpsScheme: Tariff ops scheme.
        """

        if isinstance(service, ClosestService):
            service = service.title

        plan_key = plan.strip().lower()
        service_key = service.strip().lower()

        # TODO: Костыль!
        if plan_key in self.trial_variants:
            plan_key = 'trial'

        if plan_key not in self.plan_map:
            logger.info('plan_map: %r', self.plan_map)
            logger.warning(
                'Unknown plan: «%r»; fallback to base plan', plan_key
            )

        # Fallback to base plan if plan_key is not in plan_map
        actual_plan = self.plan_map.get(plan_key, self.plan_map['base'])
        logger.info('actual_plan: %r', actual_plan)

        if service_key not in self.entity_map:
            raise KeyError("unknown service: «%s»", service)

        col: Tuple[str, str] = self.entity_map[service_key]
        logger.info('col: %r', col)
        description, title = col

        available_map = {
            plan.lower().strip(): self._parse_cell(
                self.df.at[plan, col])["available"]
            for plan in self.df.index
        }
        logger.info('available_map: %r', available_map)
        parsed = self._parse_cell(self.df.at[actual_plan, col])

        logger.info('parsed: %r', parsed)
        quota = parsed["quota"]

        return TariffOpsScheme(
            entity=title,
            actual_plan=actual_plan.strip(),
            description=description,
            available_map=available_map,
            quota=quota
        )


tariff_definer = TariffDefiner(
    Path('./src/data/service_definer/tariff_services.csv')
)
