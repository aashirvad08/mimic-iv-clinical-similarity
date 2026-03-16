**Gives personal details :**

patients.csv.gz -> subject\_id	gender	anchor\_age	anchor\_year	anchor\_year\_group

admissions.csv.gz -> subject\_id	hadm\_id	admittime	dischtime	admission\_type	admission\_location	discharge\_location 				marital\_status	race

omr.csv.gz -> subject\_id	chartdate	seq\_num	result\_name(height, weight, blood pressure, BMI)	result\_value



**Gives idea about transfers (if eventtype = is first admit then ED then it's a serious case):**  

transfers.csv.gz -> subject\_id	hadm\_id	transfer\_id	eventtype	intime	outtime	



**Gives Diagnoses and procedures for that diagnoses :** 

diagnoses\_icd.csv.gz -> subject\_id	hadm\_id	seq\_num	icd\_code	icd\_version	

d\_icd\_diagnoses.csv.gz -> icd\_code	icd\_version	long\_title

procedures\_icd.csv.gz -> subject\_id	hadm\_id	seq\_num	chartdate	icd\_code	icd\_version

d\_icd\_procedures.csv.gz -> icd\_code	icd\_version	long\_title



**Gives idea of how severe a dieasea actually is :**

labevents.csv.gz -> labevent\_id	subject\_id	hadm\_id(most are <NA>)	specimen\_id	itemid	charttime	storetime	value	valuenum	valueuom	ref\_range\_lower	ref\_range\_upper	flag	priority	comments

d\_labitems.csv.gz -> itemid	label	fluid	category



**Gives Prescription Ideas :**

prescriptions.csv.gz -> subject\_id	hadm\_id	poe\_id	poe\_seq	starttime	stoptime	drug\_type	drug	formulary\_drug\_cd	prod\_strength	form\_rx	dose\_val\_rx	dose\_unit\_rx	form\_val\_disp	form\_unit\_disp	doses\_per\_24\_hrs	route



