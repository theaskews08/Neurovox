import logging
import json
import os
import asyncio
from datetime import datetime
from pathlib import Path
from sqlitedict import SqliteDict

logger = logging.getLogger(__name__)


class MemoryManager:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = Path("../data/conversation_history/memory.sqlite")

        # Create directory if it doesn't exist
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.db_path = str(db_path)
        self.short_term_memory = []
        self.mutex = asyncio.Lock()

        # Ensure the database exists
        self._init_db()

    def _init_db(self):
        """Initialize the database"""
        with SqliteDict(self.db_path, tablename="conversations") as db:
            db.commit()

        with SqliteDict(self.db_path, tablename="entities") as db:
            db.commit()

    async def add_exchange(self, user_input, assistant_response):
        """Add a conversation exchange to memory"""
        async with self.mutex:
            timestamp = datetime.now().isoformat()

            # Add to short-term memory
            self.short_term_memory.append({
                "role": "user",
                "content": user_input,
                "timestamp": timestamp
            })

            self.short_term_memory.append({
                "role": "assistant",
                "content": assistant_response,
                "timestamp": timestamp
            })

            # Limit short-term memory to last 10 exchanges (20 messages)
            if len(self.short_term_memory) > 20:
                self.short_term_memory = self.short_term_memory[-20:]

            # Store in long-term memory
            conversation_id = timestamp[:10]  # Use date as conversation ID

            with SqliteDict(self.db_path, tablename="conversations") as db:
                if conversation_id in db:
                    conversation = db[conversation_id]
                else:
                    conversation = []

                conversation.append({
                    "user": user_input,
                    "assistant": assistant_response,
                    "timestamp": timestamp
                })

                db[conversation_id] = conversation
                db.commit()

            logger.debug(f"Added exchange to memory (conversation {conversation_id})")

    async def get_recent_context(self, max_tokens=2000):
        """Get recent conversation context for the LLM"""
        async with self.mutex:
            # This is a simplified implementation without token counting
            # In a real implementation, we would count tokens and trim accordingly
            return self.short_term_memory

    async def get_relevant_context(self, query, max_tokens=2000):
        """Get context relevant to the query"""
        # This is a simplified implementation that just returns recent context
        # In a real implementation, we would use embeddings to find relevant past conversations
        return await self.get_recent_context(max_tokens)