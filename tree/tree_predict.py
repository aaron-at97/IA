# ---- t3 ----
def read_file(file_path, data_sep=",", ignore_first_line=False):
    prototypes = []
    # Open file
    with open(file_path, "r") as fh:
        # Strip lines
        strip_reader = (line.strip() for line in fh)

        # Filter empty lines
        filtered_reader = (line for line in strip_reader if line)

        # Skip first line if needed
        if ignore_first_line:
            next(filtered_reader)

        # Split line, parse token and append to prototypes
        for line in filtered_reader:
            prototypes.append(
                [filter_token(token) for token in line.split(data_sep)]
            )

    return prototypes


def filter_token(token):
    try:
        return int(token)
    except ValueError:
        try:
            return float(token)
        except ValueError:
            return token

# ----- t4 -----
def unique_counts(part):
    #import collections
    #return dict(collections.Counter(row[-1] for row in part))
    results = {}
    for row in part:
        results[row[-1]] = results.get(row[-1], 0) + 1
    return results

# ----- t5 -----
def gini_impurity(part):
    total = float(len(part))
    results = unique_counts(part)

    return 1 - sum((count / total) ** 2 for count in results.values())

# ----- t6 -----
def entropy(part):
    from math import log
    log2 = lambda x: log(x) / log(2)
    results = unique_counts(part)
    total = float(len(part))

    return -sum(
        (count / total) * log2(count / total) for count in results.values()
    )

# ----- t7 -----
def divideset(part, column, value):
    split_function = None

    if isinstance(value, int) or isinstance(value, float):
        split_function = lambda row: row[column] >= value
    else:
        split_function = lambda row: row[column] == value

    # Split "part" according "split_function"
    set1, set2 = [], []
    for row in part:
        if split_function(row):
            set1.append(row)
        else:
            set2.append(row)

    return set1, set2

# ---- t9 ----
def buildtree(part, scoref=entropy, beta=0):

    if len(part) == 0: return decisionode()
    current_score = scoref(part)

    best_gain = 0
    best_criteria = None
    best_sets = None

    col_cont = len(part[0]) - 1
    for col in range(0, col_cont):

        column_values = {}
        for row in part:
            column_values[row[col]] = 1

        for value in column_values.keys():
            # Set 1 True y Set 2 False.
            (set1, set2) = divideset(part, col, value)
            gain = current_score - (len(set1) / len(part)) * scoref(set1) - (len(set2) / len(part)) * scoref(set2)

            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)


    if best_gain > beta:
        leftBranch = buildtree(best_sets[0])
        rightBranch = buildtree(best_sets[1])
        return decisionode(col=best_criteria[0], value=best_criteria[1], tb=leftBranch, fb=rightBranch)
    else:
        return decisionode(results=unique_counts(part))

# ---- t10 ----
def buildtree_ite(part, scoref=entropy, beta=0):
    dnode = decisionode()
    stack = [[part, dnode]]

    while 0 < len(stack):

        best_gain = 0
        best_criteria = None
        best_sets = None
        best_col = -10

        prototypes, atributo = stack.pop()

        current_score = scoref(prototypes)

        col_cont = len(part[0]) - 1
        for col in range(0, col_cont):
            criterials = []

            for row in part:
                if row[col] not in criterials:
                    criterials.append(row[col])

            for value in criterials:
                # Set 1 True y Set 2 False.
                (set1, set2) = divideset(prototypes, col, value)
                gain = current_score - (len(set1) / len(prototypes)) * scoref(set1) - (len(set2) / len(prototypes)) * scoref(set2)

                if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                    best_gain = gain
                    best_criteria = value
                    best_sets = (set1, set2)
                    best_col = col

        if best_gain > beta:

            atributo.tb = decisionode()
            atributo.fb = decisionode()
            atributo.value = best_criteria
            atributo.col = best_col
            stack.append([best_sets[0], atributo.tb])
            stack.append([best_sets[1], atributo.fb])

        else:
            atributo.results = unique_counts(prototypes)

    return dnode

# ---- t12 ----
def classify(obj, tree):

    if tree.results != None:
        return tree.results
    else:
        if isinstance(obj[tree.col], int) or isinstance(obj[tree.col], float):
            if obj[tree.col] >= tree.value:
                hoja = tree.tb
            else:
                hoja = tree.fb
        else:
            if obj[tree.col] == tree.value:
                hoja = tree.tb
            else:
                hoja = tree.fb
        return classify(obj, hoja)

def printtree(tree, indent=''):
    # Is this a leaf node?
    if tree.results is not None:
        print(indent+str(tree.results))
    else:
        # Print the criteria
        print(indent + str(tree.col)+':'+str(tree.value)+'? ')
        # Print the branches
        print(indent+'T->')
        printtree(tree.tb, indent+'  ')
        print(indent+'F->')
        printtree(tree.fb, indent+'  ')

# ----- t8 -----
class decisionode:

    def __init__(
        self, col=-1, value=None, results=None, tb=None, fb=None
    ):
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb

if __name__ == "__main__":

 prototypes = read_file("decision_tree_example.txt", data_sep=",", ignore_first_line=True)
 set1, set2 = divideset(prototypes, column=3, value=20)
 tree = buildtree(prototypes, scoref=entropy, beta=0)


 print("\n Prototipes")
 print("------------------ ")
 print(prototypes)

 print("\n Divide sets ")
 print("------------------ ")
 print ("Set 1")
 print (set1)
 print ("Set 2")
 print (set2)

 # Get a dictionary with key: class_name, value: total
 unique_counts(prototypes)

 # Get Gini impurity
 print("\n Gini Impurity")
 print("------------------ ")
 gini = gini_impurity(prototypes)
 print(gini)

 # Get entropy
 print ("\n Get entropy")
 print("------------------ ")
 entr = entropy(prototypes)
 print(entr)
 # Get BuildTree
 print ("\n Build Tree")
 print("------------------ ")
 printtree(tree)


 tree1 = buildtree_ite(prototypes, scoref=entropy, beta=0)
 print ("\n Build Tree_Ite")
 print("------------------ ")
 printtree(tree1)


