import csv
def parse_csv():
    f = open("data/Grants-20230219.csv", "r+")
    t = f.readlines()
    text = ""
    for i in t:
        text += i.replace('\0', '')
    f.seek(0)
    f.write(text)
    f.seek(0)
    csv_reader = csv.reader(f)

    # use the csv file
    papers = []
    for row in csv_reader:
        if (row == []):
            continue
        papers.append(str(row[1] + ". " + row[3]))

    f.close()
    return papers
