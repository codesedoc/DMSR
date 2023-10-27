import sys

from experiment import set_seed
from nlpx.utils import set_up_args, initialize
from nlpx.utils.runtime import RunTime

set_seed(1234)

## Specify the path of dataset
sys.argv.extend(["--data_root_dir", "storage/dataset"])
## Specify the path of checkponit of the trained model
sys.argv.extend(["--plm_name_or_path", "../models/pg2st"])
## Specify the other arguments by the json file
set_up_args("experiment/ptr/t5-args/pg2st.json")

runtime: RunTime = initialize()
pg2st_model = runtime.approach.model
pg2st_tokenizer = runtime.approach.tokenizer
input_ids = pg2st_tokenizer("The quality of this procedure is bad!", padding=True, return_tensors="pt")
preds = pg2st_model.generate(**input_ids)
generations = pg2st_tokenizer.batch_decode(preds, skip_special_tokens=True)# Error here
print(generations)