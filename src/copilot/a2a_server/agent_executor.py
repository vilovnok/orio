import logging
import uuid
import json
from typing_extensions import override


from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_task
from a2a.types import (
    DataPart,
    Part,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
    TaskStatus,
    TaskState,
    Artifact,
    Message,
    Role
)

from src.copilot.a2a_server.schemas import CopilotPayload
from src.copilot.agent import copilot_agent

from src.utils.config import push_log_context

logger = logging.getLogger(__name__)


class CopilotAgentExecutor(AgentExecutor):

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        parts = context.message.parts
        logger.debug('context: %r', context)
        if not parts:
            logger.error('No parts in message')
            raise Exception('No parts in message')

        task = context.current_task or new_task(request=context.message)

        # 1.2 Extract params from parts
        params = {}
        for part in parts:
            if hasattr(part, 'root') and isinstance(part.root, DataPart):
                params.update(part.root.data)

        logger.info(
            'params (json): \n%s',
            json.dumps(params, indent=4, ensure_ascii=False)
        )
        logger.info('type(params): %r', type(params))

        # 2. Prepare data for agent
        copilot_req = CopilotPayload(**params)
        logger.debug('copilot_req: %r', copilot_req)

        # 2.1. Processing task with context fields
        with push_log_context(
            task_id=task.id,
            task_context_id=task.context_id,
            request_id=copilot_req.request_id,
            chat_id=copilot_req.chat_id,
            account_id=copilot_req.contexts.account.id,
            user_id=copilot_req.contexts.user.id,
        ):
            # 2.2. Prepare data for agent
            state = copilot_req.prepare_data_for_agent()

            # 3. Invoke agent
            response = await copilot_agent.ainvoke(state)
            answers = response.answers
            logger.info('got %d answers', len(answers))

            for answer in answers:
                if not answer.is_knowledge_base:
                    answer.action='unknown'

            await event_queue.enqueue_event(
                TaskArtifactUpdateEvent(
                    append=False,
                    context_id=task.context_id,
                    taskId=task.id,
                    lastChunk=False,
                    # Это только конечный ответ, от тактических агентов
                    artifact=Artifact(
                        artifactId=str(uuid.uuid4()),
                        parts=[Part(root=DataPart(
                            data={'answers': answers}
                        ))]
                    )
                )
            )

            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    context_id=task.context_id,
                    taskId=task.id,
                    status=TaskStatus(
                        state=TaskState.completed,
                        # Вот это есть всегда при обновлении статуса (у всех)
                        message=Message(
                            messageId=str(uuid.uuid4()),
                            role=Role.agent,
                            parts=[Part(root=DataPart(
                                data={'answers': answers}
                            ))]
                        ),
                    ),
                    final=True
                )
            )

    @override
    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        raise Exception('cancel not supported')
