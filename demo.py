from main import parse, train, report
from src.SEPN.model import Recommender

if __name__=="__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    args = parse(print_help=True)
    train(args, Recommender)
    # report("runs")
