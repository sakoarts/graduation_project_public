import pymongo
import io
import gridfs

with open('../tools/.mongo', 'r') as f:
    auth_url = f.read()
    client = pymongo.MongoClient(auth_url)

def cout(_id):
    df = io.StringIO(r.find({'_id': _id})[0]['captured_out']).read()
    return df

def experiment_name(_id):
    df = r.find({'_id': _id})[0]['experiment']['name']
    return df

def runs():
    return db.get_collection('runs')

def run(_id):
    return r.find({'_id': _id})[0]

def config(_id):
    df = run(_id)['config']
    return df

def result(_id):
    df = run(_id)['result']
    return df

def source(_id, print_files=False):
    fs = gridfs.GridFS(db)
    ### load keras model from json:
    files = db['fs.files']
    r_ = list(r.find({"_id": _id}))[0]
    source_files = r_['experiment']['sources']
    files = {}
    for sf in source_files:
        file_id = sf[-1]
        s = fs.get(file_id).read()
        files[sf[0]] = s.decode()
        if print_files:
            print(files[sf[0]])
            print('\n\n\n')
    return files

db = client.graduation
r = runs()
