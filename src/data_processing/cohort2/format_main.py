from src.data_processing.cohort2.format_lesions import main as format_lesions
from src.data_processing.cohort2.format_deficits import main as format_deficits
from src.data_processing.cohort2.format_dems import main as format_dems

if __name__ == '__main__':
    format_lesions()
    format_deficits()
    format_dems()