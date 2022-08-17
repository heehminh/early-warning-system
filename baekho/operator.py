from apscheduler.schedulers.background import BackgroundScheduler
from django_apscheduler.jobstores import register_events, DjangoJobStore
from .views import update_csv

def start():
    scheduler=BackgroundScheduler(timezone="Asia/Seoul")
    scheduler.add_job(update_csv, "cron", minute=28)
    print("==크롤링 스케줄러 시작==")
    scheduler.start()

    print("==크롤링 스케줄러 완료==")