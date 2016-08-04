# coding: utf-8

import glob
import sys
import os
import csv

filedir = sys.argv[2]
outfilename = sys.argv[1]

dup_check = dict()
with open(outfilename, 'wt', encoding='utf-8') as f:
  filelist = glob.glob( os.path.join(filedir,"*.csv"))
  csv_writer = csv.writer(f, delimiter=',')
  for filename in filelist:
    try:
      with open(filename, 'rt', encoding='cp949') as fp:
        fp.readline()
        for row in fp.readlines():
          fields = row.split(',', 1)
          strings = fields[1].strip()
          if strings[-1] == ',':
            strings = strings[1:-1]
          if strings not in dup_check:
            keywords = strings.split('|')
            if len(keywords) < 2:
              print ("error keywords={}", keywords)
              continue
            try:
              csv_writer.writerow(keywords)
              dup_check[strings] = 1
            except Exception as e:
              print ("except:{}, error={},keywords={}".format(filename, e, keywords))
    except Exception as e:
      print ("except:{}, error={}".format(filename, e))

