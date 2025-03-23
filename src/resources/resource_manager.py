import logging
import asyncio

logger = logging.getLogger(__name__)


class NeuralResourceManager:
    def __init__(self, gpu_memory=8192):  # Memory in MB
        self.total_memory = gpu_memory
        self.allocations = {}
        self.priorities = {
            "stt": 1,  # Higher priority components
            "llm": 0,
            "tts": 2,
            "classifier": 3
        }
        self.min_allocations = {
            "stt": 1024,  # Minimum memory requirements
            "llm": 3072,
            "tts": 512,
            "classifier": 256
        }
        self.mutex = asyncio.Lock()

    async def allocate(self, component, requested_memory=None):
        """Allocate memory for a component"""
        async with self.mutex:
            # Get minimum required memory
            min_memory = self.min_allocations.get(component, 256)

            # If requested memory is None, use minimum
            if requested_memory is None:
                requested_memory = min_memory

            logger.debug(f"Allocating {requested_memory}MB for {component}")

            # Check current allocations
            current_usage = sum(self.allocations.values())
            available = self.total_memory - current_usage

            if requested_memory <= available:
                # We can allocate without adjustments
                self.allocations[component] = requested_memory
                return requested_memory

            # Need to optimize allocations
            logger.info(f"Optimizing memory allocations to fit {component}")
            return await self._optimize_for_component(component, requested_memory)

    async def release(self, component):
        """Release memory allocated to a component"""
        async with self.mutex:
            if component in self.allocations:
                freed = self.allocations.pop(component)
                logger.debug(f"Released {freed}MB from {component}")
                return freed
            return 0

    async def _optimize_for_component(self, target_component, requested_memory):
        """Optimize memory allocations to make room for a component"""
        # Sort components by priority (higher number = lower priority)
        sorted_components = sorted(
            self.allocations.keys(),
            key=lambda x: self.priorities.get(x, 99)
        )

        # Start taking memory from lowest priority components
        needed = requested_memory - (self.total_memory - sum(self.allocations.values()))

        if needed <= 0:
            # Should not happen, but just in case
            self.allocations[target_component] = requested_memory
            return requested_memory

        # Take memory from other components
        for component in reversed(sorted_components):
            # Skip if this is the target component
            if component == target_component:
                continue

            # Get current allocation and minimum
            current = self.allocations[component]
            minimum = self.min_allocations.get(component, 256)

            # How much can we take?
            can_take = max(0, current - minimum)

            if can_take > 0:
                # Take what we need, up to what's available
                to_take = min(needed, can_take)
                self.allocations[component] -= to_take
                needed -= to_take

                logger.info(f"Reduced {component} allocation by {to_take}MB")

                if needed <= 0:
                    break

        # If we still need memory, we have to reduce to bare minimum
        if needed > 0:
            logger.warning(
                f"Could not fully optimize for {target_component}, allocating {requested_memory - needed}MB instead of {requested_memory}MB")
            self.allocations[target_component] = requested_memory - needed
            return requested_memory - needed

        # Allocate the requested memory
        self.allocations[target_component] = requested_memory
        return requested_memory

    def get_allocation(self, component):
        """Get current allocation for component"""
        return self.allocations.get(component, 0)

    def get_available_memory(self):
        """Get total available memory"""
        return self.total_memory - sum(self.allocations.values())