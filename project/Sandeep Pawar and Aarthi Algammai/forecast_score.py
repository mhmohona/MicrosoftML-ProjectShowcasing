

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType


def init():

    global model
    
    model = os.path.join(os.getenv('MODEL_DIR'), 'forecast.py')
    

    input_sample = pd.DataFrame({'Fuel':,['a'] * 10
                                 'Sales': np.arange(1,11, dtype='float64').tolist()}, 
                                  index=pd.date_range('1-1-2000', freq='MS',periods=10))


    output_sample = pd.DataFrame({'Pellet'    : np.arange(1,11, dtype='float64').tolist(), 
                                  'Firewood'  : np.arange(1,11, dtype='float64').tolist(),
                                  'Briquette' : np.arange(1,11, dtype='float64').tolist()})
    
    

@input_schema('data', PandasParameterType(input_sample, enforce_shape=False))
@output_schema(PandasParameterType (output_sample))

data_path   = "https://powerbibyos3.blob.core.windows.net/powerbi/Trial/test/Query.csv.snapshots/data.csv"
result_path = "https://powerbibyos3.blob.core.windows.net/powerbi/Trial/result/"



def run(data):
    try:
    
        
        result = pd.DataFrame({ 
                                 'Pellet': pellet_forecast(data.query('Fuel == "Pellet"')['Sales'],12).tolist(),
                                 'Firewood': firewood_forecast(data.query('Fuel == "Firewood"')['Sales'],12).tolist(),
                                 'Briquette': briquette_forecast(data.query('Fuel == "Briquette"')['Sales'],12).tolist()
                             })
        
        return result.to_csv(result_path, sep='\t')
    
    except Exception as e:
        error = str(e)
        return error
