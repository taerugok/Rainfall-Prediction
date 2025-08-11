import datetime
import os

#set training end date to 1 week ago
end_date=int((datetime.date.today()-datetime.timedelta(days=7)).strftime('%Y%m%d'))

#go back one month because for some reason knmi website does not provide the current month data
#since this is just to try github action capabilities, and not make real time predcitions that will be used, it s ok
#better than causing data leakage 
#it will simulate how system would work one month ago
end_date=(datetime.date.today()-datetime.timedelta(days=37))

#start_date is 10 years before end_date
start_date=end_date-datetime.timedelta(days=3650)

end_date=int(end_date.strftime('%Y%m%d'))
start_date=int(start_date.strftime('%Y%m%d'))

#note that here there are complications due to leap year, months that are not 30 days etc but it is not so significant for the purposes of this project

var_name='TRAIN_START_DATE'
var_value=str(start_date)
echo_statement="echo "+"\""+var_name+"="+var_value+"\""+" >> $GITHUB_ENV"
print("sending the following to os.system")
print(echo_statement)
os.system(echo_statement)

var_name='TRAIN_END_DATE'
var_value=str(end_date)
echo_statement="echo "+"\""+var_name+"="+var_value+"\""+" >> $GITHUB_ENV"
print("Sending the following to os.system")
print(echo_statement)
os.system(echo_statement)
