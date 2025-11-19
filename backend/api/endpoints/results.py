"""Results API Endpoint"""
from fastapi import APIRouter

router = APIRouter()

@router.get("/dashboard/stats")
async def get_dashboard_stats():
    """Get dashboard statistics"""
    return {
        'total_processed': 45230,
        'processed_today': 1250,
        'in_progress': 73,
        'parallel_workers': 10,
        'success_rate': 99.2,
        'profit_margin': 58.0,
        'total_revenue': 361840.00,
        'total_cost': 151966.00,
        'total_profit': 209874.00,
        'avg_cost_per_video': 3.36,
        'days_to_1m': 542,
        'recent_batches': [],
        'top_failure_types': ['Counting', 'Temporal', 'Context']
    }