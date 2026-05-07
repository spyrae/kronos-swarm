"""Entry point: python -m kronos"""

import asyncio
import logging

from dotenv import load_dotenv

load_dotenv()

from kronos.app import main
from kronos.logging import install_pii_filter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
install_pii_filter()

if __name__ == "__main__":
    asyncio.run(main())
