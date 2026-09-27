import asyncio
import typer
from src.utils.config import inited_config
from src.copilot.provider.llm import LLMProvider, create_llm

import logging


# app = typer.Typer(add_completion=False)
# @app.command()

logger = logging.getLogger(__name__)

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoxLCJhY2NvdW50X2lkIjoxLCJzZXJ2aWNlIjoibG9jYWwtZGV2IiwiZXhwIjoxNzkwNDYyNDMwfQ.20v42Ke0z-ql0CSL_DRwzC1W79kCP-UiNZMXKtSMHlY"



def main():
    setup_logging()
    try:
        logger.info("Starting Copilot worker")


        
    except KeyboardInterrupt:
        logger.info("Work finished by user request")


if __name__ == "__main__":
    main()
