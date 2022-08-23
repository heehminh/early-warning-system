from apscheduler.schedulers.background import BackgroundScheduler
from django_apscheduler.jobstores import register_events, DjangoJobStore
from .views import save_last, send_mail, china, usa, japan, vietnam, australia

def start():
    scheduler=BackgroundScheduler(timezone="Asia/Seoul")

    scheduler.add_job(save_last, "cron", hour=14, minute=54)

    hour =14
    minute=55
    # 5개국
    scheduler.add_job(vietnam, "cron", hour= hour, minute=minute)
    scheduler.add_job(usa, "cron", hour= hour, minute=minute+1)
    scheduler.add_job(japan, "cron", hour= hour, minute=minute+2)
    scheduler.add_job(china, "cron", hour= hour, minute=minute+3)
    scheduler.add_job(australia, "cron", hour= hour, minute=minute+4)

    scheduler.add_job(send_mail, "cron", hour=15, minute=15)

    print("==크롤링 스케줄러 시작==")
    scheduler.start()
    print("==크롤링 스케줄러 완료==")