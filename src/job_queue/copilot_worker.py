import logging

import json
from datetime import datetime
from typing import Any

from pystalkd.Beanstalkd import SocketError

from src.copilot.agent import copilot_agent
from src.copilot.services.usage.api import send_usage_tracking

from src.job_queue.schemes import CopilotRequest, CopilotResponse
from job_queue.processing import ConnectionManager, JobFetcher

from src.utils.config import log_context, inited_config as config

logger = logging.getLogger(__name__)


class JobProcessor:
    """Processing one job:
    from receiving to publishing results and tracking."""
    def __init__(self, conn_mgr: ConnectionManager):
        self.conn_mgr = conn_mgr

    async def process(self, job: Any) -> None:
        token = None
        try:
            logger.info("Processing job_id=%s", job.job_id)
            logger.debug("Job body: %s", job.body)
            payload = json.loads(job.body)

            logger.debug("job_payload: %s", payload)
            request = CopilotRequest(**payload['workload'])
            logger.debug("job_request: %s", request)

            token = log_context.set({
                "job_id": job.job_id,
                "request_id": request.request_id,
                "chat_id": request.chat_id,
                "account_id": request.contexts.account.id,
                "user_id": request.contexts.user.id,
            })

            state = request.prepare_data_for_agent()

            logger.debug("prepared_state: %r", state)
            logger.info("Calling agent for thread_id=%s", state.thread_id)

            start = datetime.now()

            response = await copilot_agent.ainvoke(state)

            elapsed = (datetime.now() - start).total_seconds()
            logger.info("Agent returned in %.2f sec: %s", elapsed, response)

            await self._publish_results(request, response)
            job.delete()

            await self._send_usage(response.usage, request)

        except Exception as e:
            logger.error("Error processing job: %s", e, exc_info=True)
            logger.warning(
                "Deleting failed job %s", getattr(job, 'job_id', None)
            )
            try:
                job.delete()
            except Exception:
                logger.error("Failed to delete job", exc_info=True)
        finally:
            if token:
                log_context.reset(token)

    async def _publish_results(
        self,
        request: CopilotRequest,
        response: Any
    ) -> None:

        copilot_resp = CopilotResponse(
            request_id=request.request_id,
            chat_id=request.chat_id,
            contexts=request.contexts,
            question=request.question,
            answers=response.answers,
            enrich=request.enrich,
            sla=request.sla,
        )
        payload = {'workload': json.loads(copilot_resp.model_dump_json())}

        client = await self.conn_mgr.get_client()
        try:
            client.put(json.dumps(payload))
        except (SocketError, ValueError) as e:
            logger.error("Failed to publish results: %s", e, exc_info=True)
            self.conn_mgr.reset()
            client = await self.conn_mgr.get_client()
            client.put(json.dumps(payload))
        except Exception as e:
            logger.error("Failed to publish results: %s", e, exc_info=True)

        logger.debug("Published result for request %s", request.request_id)

    async def _send_usage(self, usage: Any, request: CopilotRequest) -> None:
        try:
            await send_usage_tracking(
                usage_by_model=usage,
                account_id=request.contexts.account.id,
                user_id=request.contexts.user.id,
                group=request.contexts.user.group
            )
        except Exception as e:
            logger.error("Usage tracking failed: %s", e, exc_info=True)


class CopilotWorker:
    """Главный класс-воркер."""
    def __init__(self):
        self.conn_mgr = ConnectionManager(
            host=config.beanstalkd.host,
            port=config.beanstalkd.port,
            watch_tube=config.beanstalkd.jobs_queue,
            use_tube=config.beanstalkd.results_queue,
        )
        self.fetcher = JobFetcher(self.conn_mgr)

    async def run(self) -> None:
        while True:
            job = await self.fetcher.fetch()
            if not job:
                continue
            processor = JobProcessor(self.conn_mgr)
            await processor.process(job)
