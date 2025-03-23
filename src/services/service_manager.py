import asyncio
import logging
from typing import Dict, List, Callable, Optional

logger = logging.getLogger(__name__)


class ServiceManager:
    def __init__(self):
        self.services = {}
        self.health_checks = {}
        self.dependencies = {}
        self.restart_attempts = {}
        self.max_restart_attempts = 3
        self.is_running = False
        self.monitor_task = None

    def register_service(self, name, start_fn, stop_fn, health_check_fn=None, dependencies=None):
        """Register a service with the manager"""
        self.services[name] = {
            "start": start_fn,
            "stop": stop_fn,
            "status": "stopped"
        }

        if health_check_fn:
            self.health_checks[name] = health_check_fn

        if dependencies:
            self.dependencies[name] = dependencies
        else:
            self.dependencies[name] = []

        self.restart_attempts[name] = 0
        logger.info(f"Registered service: {name}")

    async def start_all(self):
        """Start all services in dependency order"""
        self.is_running = True

        # Build dependency tree
        dependency_tree = self._build_dependency_tree()
        logger.info(f"Starting services in order: {dependency_tree}")

        # Start services in order
        for service_name in dependency_tree:
            logger.info(f"Starting service: {service_name}")
            success = await self._start_service(service_name)
            if not success:
                logger.error(f"Failed to start service: {service_name}")
                # If a critical service fails, we might want to handle that here

        # Start health check monitoring
        self.monitor_task = asyncio.create_task(self._health_monitor())
        logger.info("All services started, health monitoring active")

    async def stop_all(self):
        """Stop all services in reverse dependency order"""
        self.is_running = False

        # Cancel health monitor
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        # Build dependency tree and reverse it
        dependency_tree = self._build_dependency_tree()
        dependency_tree.reverse()

        logger.info(f"Stopping services in order: {dependency_tree}")

        # Stop services in reverse order
        for service_name in dependency_tree:
            if self.services[service_name]["status"] == "running":
                logger.info(f"Stopping service: {service_name}")
                try:
                    await self.services[service_name]["stop"]()
                    self.services[service_name]["status"] = "stopped"
                except Exception as e:
                    logger.error(f"Error stopping service {service_name}: {str(e)}")

    async def _start_service(self, name):
        """Start a single service"""
        if name not in self.services:
            logger.error(f"Service not registered: {name}")
            return False

        if self.services[name]["status"] == "running":
            logger.info(f"Service already running: {name}")
            return True

        # Check dependencies
        for dep in self.dependencies.get(name, []):
            if self.services[dep]["status"] != "running":
                logger.error(f"Dependency {dep} not running for service {name}")
                return False

        try:
            success = await self.services[name]["start"]()
            if success:
                self.services[name]["status"] = "running"
                self.restart_attempts[name] = 0
                logger.info(f"Service {name} started successfully")
                return True
            else:
                logger.error(f"Service {name} start function returned False")
                self.services[name]["status"] = "failed"
                return False
        except Exception as e:
            logger.error(f"Failed to start service {name}: {str(e)}")
            self.services[name]["status"] = "failed"
            return False

    async def _health_monitor(self):
        """Monitor service health and restart failed services"""
        logger.info("Health monitoring started")

        while self.is_running:
            for name, check_fn in self.health_checks.items():
                if self.services[name]["status"] == "running":
                    try:
                        is_healthy = await check_fn()
                        if not is_healthy:
                            logger.warning(f"Service {name} is unhealthy, attempting restart")
                            await self._restart_service(name)
                    except Exception as e:
                        logger.error(f"Health check for {name} failed: {str(e)}")
                        await self._restart_service(name)

            # Wait before next check
            await asyncio.sleep(30)

    async def _restart_service(self, name):
        """Attempt to restart a failed service"""
        if self.restart_attempts[name] >= self.max_restart_attempts:
            logger.error(f"Service {name} failed too many times, not restarting")
            self.services[name]["status"] = "failed"
            return False

        try:
            # Stop the service if it's running
            if self.services[name]["status"] == "running":
                await self.services[name]["stop"]()

            # Restart it
            success = await self.services[name]["start"]()
            if success:
                self.services[name]["status"] = "running"
                self.restart_attempts[name] += 1
                logger.info(f"Service {name} restarted successfully (attempt {self.restart_attempts[name]})")
                return True
            else:
                logger.error(f"Service {name} restart failed")
                self.services[name]["status"] = "failed"
                return False
        except Exception as e:
            logger.error(f"Failed to restart service {name}: {str(e)}")
            self.services[name]["status"] = "failed"
            return False

    def _build_dependency_tree(self):
        """Build a flat list of services in dependency order"""
        visited = set()
        order = []

        def visit(name):
            if name in visited:
                return
            visited.add(name)
            for dep in self.dependencies.get(name, []):
                visit(dep)
            order.append(name)

        for service in self.services:
            visit(service)

        return order