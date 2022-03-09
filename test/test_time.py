from datetime import datetime, timezone, timedelta

import pytz

if __name__ == '__main__':
    pytz.timezone('Asia/Shanghai')

    SHA_TZ = timezone(
        timedelta(hours=8),
        name='Asia/Shanghai',
    )

    # 协调世界时
    utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
    print("UTC:")
    print(utc_now, utc_now.time())
    print(utc_now.date(), utc_now.tzname())

    # 北京时间
    beijing_now = utc_now.astimezone(SHA_TZ)
    print("Beijing:")
    print(beijing_now.strftime('%Y-%m-%d-%H-%M'))
    print(beijing_now, beijing_now.time())
    print(beijing_now.date(), beijing_now.tzname())

    # 系统默认时区
    local_now = utc_now.astimezone()
    print("Default:")
    print(local_now, local_now.time())

    print(local_now.date(), local_now.tzname())

