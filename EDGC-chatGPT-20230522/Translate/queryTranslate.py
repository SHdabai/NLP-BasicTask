import pycorrector
from dict import Dict



def translate(query):
    _ = Dict([query])
    a = _.parse()

    return a


if __name__=="__main__":
    pass