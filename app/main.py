from fastapi import FastAPI, BackgroundTasks
from coordinator import TradingCoordinator
from loguru import logger

app = FastAPI(title="MNNIT Stock AI Agent")
coordinator = TradingCoordinator()

@app.get("/")
def health_check():
    """Confirms the container is alive."""
    return {"status": "online", "mode": "production"}

@app.post("/run-cycle")
async def trigger_cycle(background_tasks: BackgroundTasks):
    """
    Triggered by GCP Cloud Scheduler. 
    Runs the trade cycle in the background to prevent HTTP timeouts.
    """
    logger.info("📡 External trigger: Starting Market Analysis Cycle...")
    
    # We use BackgroundTasks so the HTTP response returns immediately (200 OK)
    # while the bot works in the background.
    background_tasks.add_task(coordinator.run_cycle)
    
    return {"message": "Trading cycle initiated"}