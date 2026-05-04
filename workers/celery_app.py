from celery import Celery

from config.settings import REDIS_URL

app = Celery("contradiction_pipeline", broker=REDIS_URL, backend=REDIS_URL)
app.conf.task_serializer = "json"
app.conf.result_serializer = "json"
app.conf.task_acks_late = True
app.conf.task_reject_on_worker_lost = True
app.conf.worker_prefetch_multiplier = 1
