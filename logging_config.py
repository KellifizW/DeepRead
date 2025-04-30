import logging
from datetime import datetime
import pytz

HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")

class HongKongFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=HONG_KONG_TZ)
        return dt.strftime(datefmt or "%Y-%m-%d %H:%M:%S,%f")[:-3] + " HKT"

def configure_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    formatter = HongKongFormatter("%(asctime)s - %(levelname)s - %(message)s")
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
