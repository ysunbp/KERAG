from datetime import datetime, timedelta

def parse_date_range(date_list):
    date_format = "%Y-%m-%d"
    dates = []

    for item in date_list:
        if "EST" in item:
            item = item[:-13]
        if len(item) == 4 and item.isdigit():  # 处理年份
            year = int(item)
            dates.append(year)
        else:  # 处理具体日期
            current_date = datetime.strptime(item, date_format)
            dates.append(current_date)

    # 如果有两个日期，生成日期范围
    if len(dates) == 2 and all(isinstance(date, datetime) for date in dates):
        start_date, end_date = sorted(dates)
        dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    
    # 返回结果
    if all(isinstance(date, int) for date in dates):  # 如果全是年份
        if len(dates) == 1:
            return [str(date) for date in dates]  # 去重后返回年份（排序）
        else:
            return list(map(str, range(dates[0], dates[1]+1)))
    else:  # 如果有日期
        return [date.strftime(date_format) for date in dates]

# 示例
print(parse_date_range(["2024-02-21 00:00:00 EST", "2024-02-28 00:00:00 EST"]))  # 返回 ['2024-03-10', '2024-03-
#print(parse_date_range(["2024-03-09"]))                 # 返回 ['2024-03-09']
#print(parse_date_range(["2024"]))                        # 返回 [2024]
#print(parse_date_range(["2023", "2029"]))  
#print(parse_date_range([]))  