import pandas as pd
import matplotlib.pyplot as plt

goal_distance = 3250

def meanAndStd(file):
	df = pd.read_csv(file)
	df.fillna(0, inplace=True)

	#df.loc[df["distance"] > 1000, "start_from"] = 1
	#df = df[~((df["distance"] >= 3000) & (df["distance"] <= 3100))]

	start_from = 0
	life_spent = 0
	arrive_forty = 0
	spawn_from_forty_dis = 1100

	for index, row in df.iterrows():
		if(start_from == 0):
			if(row["start_from"] == 1):
				start_from = 1
				arrive_forty = arrive_forty + 1
			elif(row["distance"] >= spawn_from_forty_dis):
				arrive_forty = arrive_forty + 1
			else:
				life_spent = life_spent + 1
		else:
			if(row["start_from"] == 0):
				start_from = 0

	print(life_spent, arrive_forty)

	"""
	is_end = 0
	temp_life_spent_to_end = 0
	life_spent_to_end = 0
	arrive_end = 0
	for index, row in df.iterrows():
		if(row["distance"] >= 3200):
			arrive_end = arrive_end + 1
			life_spent_to_end = life_spent_to_end + temp_life_spent_to_end
			temp_life_spent_to_end = 0
		else:
			temp_life_spent_to_end = temp_life_spent_to_end + 1
	"""
	
	return df, life_spent, arrive_forty

plotDF = pd.DataFrame()
firstData = True
mode = "fine_tuned/"
level = "1-2/bt/"
iterations = ""
file = "BT_"
fileName = [
	"distance/" + mode + level + iterations + file + "0.csv",
	"distance/" + mode + level + iterations + file + "1.csv",
	"distance/" + mode + level + iterations + file + "2.csv",
	"distance/" + mode + level + iterations + file + "3.csv",
	#"distance/" + mode + level + iterations + file + "4.csv",
	#"distance/" + mode + level + iterations + "test_distance_bonus33_0.csv",
	#"distance/" + mode + level + iterations + "test_distance_bonus33_1.csv",
	#"distance/" + mode + level + iterations + "test_distance_bonus33_2.csv",
	#"distance/" + mode + level + iterations + "test_distance_bonus33_3.csv",
	#"distance/" + mode + level + iterations + "test_distance_bonus33_4.csv",
]

life_spent = 0
arrive_forty = 0
for f in fileName:
	temp, temp_life_spent, temp_arrive_forty = meanAndStd(f)
	life_spent = life_spent + temp_life_spent
	arrive_forty = arrive_forty + temp_arrive_forty

	if(firstData):
		firstData = False
		plotDF = temp
	else:
		plotDF = plotDF.append(temp, ignore_index=True)

print(plotDF.shape)
print("from zero mean:{0}".format(round(plotDF[plotDF["start_from"] == 0]["distance"].mean(), 2)))
print("from zero std:{0}".format(round(plotDF[plotDF["start_from"] == 0]["distance"].sem(), 2)))
print("from zero max:%f" % plotDF[plotDF["start_from"] == 0]["distance"].max())
print("from zero count:%f" % plotDF[plotDF["start_from"] == 0]["distance"].count())
print("")

print("from forty mean:{0}".format(round(plotDF[plotDF["start_from"] == 1]["distance"].mean(), 2)))
print("from forty std:{0}".format(round(plotDF[plotDF["start_from"] == 1]["distance"].sem(), 2)))
print("from forty max:%f" % plotDF[plotDF["start_from"] == 1]["distance"].max())
print("from forty count:%f" % plotDF[plotDF["start_from"] == 1]["distance"].count())
print("")

dis_bigger_200 = 100 * plotDF.query('distance > 200 & start_from == 0').count()[0] / float(plotDF[plotDF["start_from"] == 0].count()[0])
dis_bigger_400 = 100 * plotDF.query('distance > 400 & start_from == 0').count()[0] / float(plotDF[plotDF["start_from"] == 0].count()[0])
dis_bigger_600 = 100 * plotDF.query('distance > 600 & start_from == 0').count()[0] / float(plotDF[plotDF["start_from"] == 0].count()[0])
print("% distance > 200:{0}".format(round(dis_bigger_200, 2)))
print("% distance > 400:{0}".format(round(dis_bigger_400, 2)))
print("% distance > 600:{0}".format(round(dis_bigger_600, 2)))
print("")

if(arrive_forty > 0):
	print("spent %f lives to arrive forty percent." % (life_spent / float(arrive_forty)))

#print(plotDF.count())
#plotDF = plotDF[~((plotDF["distance"] >= 3000) & (plotDF["distance"] <= 3100))]
#print(plotDF.count())
#arrive_end = plotDF["distance"].count() / float(plotDF[plotDF["distance"] >= goal_distance]["distance"].count())
print(plotDF[plotDF["distance"] < goal_distance]["distance"].count())
print(plotDF[plotDF["distance"] >= goal_distance]["distance"].count())
deadTimes = plotDF[plotDF["distance"] < goal_distance]["distance"].count()
goalTime = float(plotDF[plotDF["distance"] >= goal_distance]["distance"].count())
arrive_end =  deadTimes / goalTime
if(arrive_end > 0):
	print("spent %f lives to arrive goal." % (arrive_end))

#a = plotDF[plotDF["distance"] >= goal_distance]["distance"].count()
#b = plotDF[((plotDF["distance"] >= 3000) & (plotDF["distance"] <= 3100))]["distance"].count()
#print("{0} / {1}".format(a, b))

#plotDF.plot()
#plt.show()
