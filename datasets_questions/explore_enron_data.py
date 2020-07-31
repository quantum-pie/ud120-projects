#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

print(len(enron_data))
print(len(enron_data.values()[0]))

interesting_stuff = {k: v for k, v in enron_data.items() if v["poi"] == 1}
print(len(interesting_stuff))

print(enron_data["PRENTICE JAMES"]["total_stock_value"])
print(enron_data["COLWELL WESLEY"]["from_this_person_to_poi"])
print(enron_data["SKILLING JEFFREY K"]["exercised_stock_options"])

print(enron_data["SKILLING JEFFREY K"]["total_payments"])
print(enron_data["LAY KENNETH L"]["total_payments"])
print(enron_data["FASTOW ANDREW S"]["total_payments"])

with_salary = {k: v for k, v in enron_data.items() if v["salary"] != 'NaN'}
print(len(with_salary))

with_email = {k: v for k, v in enron_data.items() if v["email_address"] != 'NaN'}
print(len(with_email))

no_total_payments = {k: v for k, v in enron_data.items() if v["total_payments"] == 'NaN'}
print(len(no_total_payments))
print(len(no_total_payments) / float(len(enron_data)))

no_total_payments_interesting = {k: v for k, v in interesting_stuff.items() if v["total_payments"] == 'NaN'}
print(len(no_total_payments_interesting))
print(len(no_total_payments_interesting) / float(len(interesting_stuff)))


