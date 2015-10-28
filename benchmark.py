import cPickle
import numpy
import pandas
from collections import Counter
import xgboost as xgb
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Transform data
print "Transforming dates into numeric..."
weekdays = {"Sunday": 0,
            "Monday": 1,
            "Tuesday": 2,
            "Wednesday": 3,
            "Thursday": 4,
            "Friday": 5,
            "Saturday": 6}

#### Train
# Read training data
train_file = "train.csv"
print "Reading", train_file
train = pandas.read_csv(train_file)

# Convert all Dates to numbers and Descriptions to strings
train.DepartmentDescription = map(lambda a: str(a), train.DepartmentDescription)
train.Weekday = map(lambda a: weekdays[a], train.Weekday)

# Count visits
c = Counter(train.VisitNumber)
train["VisitCount"] = map(lambda a: c[a], train.VisitNumber)

# Perform tf-idf
print "Creating ngrams...",
ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(5, 5), min_df=1)
counts = ngram_vectorizer.fit_transform(train.DepartmentDescription)
print "Done"
print "Performing tfidf...",
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(counts)
print "Done"

# Create train matrix
print "Creating train X and y matrics...",
X = hstack((train[train.columns - ["VisitNumber", "TripType", "DepartmentDescription"]].to_sparse(), tfidf))
y = train.TripType
print "Done"

# Save features
train_features = "train_features.pkl"
print "Saving train features to", train_features, "...",
with open(train_features, "wb") as output:
    cPickle.dump((X, y), output)
print "Done"

#### Build Model
xgb_params = {"objective": "multi:softprob",
              "eta": 0.8,
              "max_depth": 10,
              "min_child_weight": 4,
              "subsample": 0.7,
              "colsample_bytree": 0.7,
              "num_class": 38,
              "seed": 1}
xgb_num_trees=200

print "Transforming labels to [0...38]"
sample_files = "sample_submission.csv"
print "Reading", sample_files
samples = pandas.read_csv(sample_files)
sample_cols = samples.columns[1:]
labels = map(lambda a: int(a.split("_")[-1]), sample_cols)
labels_dict = dict(zip(labels, range(len(labels))))
print "new labels are: ", labels_dict
y_train = map(lambda a: labels_dict[a], y)

print "Training model..."
model = xgb.train(xgb_params, xgb.DMatrix(X, y_train), xgb_num_trees)
model_file = "model.pkl"
print "Saving model to", model_file, "..." 
with open(model_file, "wb") as output:
    cPickle.dump(model, output)

#### Prediction
# Perform same transformation to test
# Read test data
test_file = "test.csv"
print "Reading", test_file
test = pandas.read_csv(test_file)

# Convert all Dates to numbers and Descriptions to strings
test.DepartmentDescription = map(lambda a: str(a), test.DepartmentDescription)
test.Weekday = map(lambda a: weekdays[a], test.Weekday)

# Count visits
c = Counter(test.VisitNumber)
test["VisitCount"] = map(lambda a: c[a], test.VisitNumber)

# Perform tf-idf
print "Creating ngrams...",
test_counts = ngram_vectorizer.transform(test.DepartmentDescription)
print "Done"
print "Performing tfidf...",
test_tfidf = transformer.fit_transform(test_counts)
print "Done"

# Create test matrix
print "Creating test X matrix...",
test_X = hstack((test[test.columns - ["VisitNumber", "DepartmentDescription"]].to_sparse(), test_tfidf))
print "Done"

# Save test features
test_features = "test_features.pkl"
print "Saving test features to", test_features, "...",
with open(test_features, "wb") as output:
    cPickle.dump(test_X, output)
print "Done"

# Predict using test data
print "Making Predictions..."
preds = model.predict(xgb.DMatrix(test_X))
print "Done"

# Average and dump the output
visit_numbers = map(lambda a: int(a), numpy.array(test.VisitNumber).reshape(len(test.VisitNumber), 1))
intermediate_preds = numpy.hstack((visit_numbers, preds))
gps = intermediate_preds.groupby("VisitNumber")
final = pandas.DataFrame(columns = samples.columns)

for d in gps.groups.keys():
    req_rows = intermediate_preds.ix[gps.groups[d]]
    means = req_rows.mean(axis=0)
    means["VisitNumber"] = str(int(means["VisitNumber"]))
    final.loc[len(final)] = means

final_output = "benchmark.csv"
print "Dumping results to", final_output
final.to_csv(final_output, index=False)
