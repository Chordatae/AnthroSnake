import pandas as pd
import csv
from datetime import datetime
import logging
from typing import Optional, List, Literal, Dict, Tuple
from IPython.display import display

"""
This program calculates Weight-for-Age (WFA), Height-for-Age (HFA), and BMI-for-Age (BMIFA) Z-scores for participants 
aged 0 to 19 years (WFA up to 10 years) based on the provided raw data and standard LMS tables.

Based on the methodology described in:
Computation of Centiles and Z-Scores for Height-for-Age, Weight-for-Age and BMI-for-Age
published by the World Health Organization (WHO).

Features:
- Imports standard LMS tables for boys and girls to use as reference datasets.
- Accepts raw data from a CSV file with specific columns: number (optional identifier), birthday, sex, height, weight, 
  date_of_visit, and age_months (optional if birthday and date_of_visit are provided).
- Handles missing data and performs necessary data transformations.
- Computes the Z-scores for HFA, WFA, and BMIFA for each participant.
- Outputs the results in a structured DataFrame, with the option to save as a CSV file.

Requirements:
- The raw data CSV file should have columns with headers 'number', 'birthday', 'sex', 'height', 'weight', 
  'date_of_visit', and 'age_months'. The order of columns and the presence of other extraneous columns is unimportant.
- All date entries should be in the MM/DD/YYYY format. Height entries should be in centimeters. Weight entries should be in kilograms.
- The LMS tables should be available in the specified directory and format.

Usage:
To use this program, provide the raw data CSV file's name when prompted. After processing, the program will display 
the computed Z-scores in a tabular format and offer an option to save them to a new CSV file.
"""

# Constants
DAYS_PER_MONTH = 30.4375
MAX_AGE_MONTHS_FOR_CALC = 228
MAX_AGE_MONTHS_FOR_WFA = 120
LMS_FILE_NAMES = {
    'wfa': ("wfa_boys_0_to_10_years_LMS.csv", "wfa_girls_0_to_10_years_LMS.csv"),
    'hfa': ("hfa_boys_0_to_19_years_LMS.csv", "hfa_girls_0_to_19_years_LMS.csv"),
    'bmifa': ("BMIfa_boys_0_to_19_years_LMS.csv", "BMIfa_girls_0_to_19_years_LMS.csv")
}
DEFAULT_OUTPUT_NAME = "output_with_z_scores.csv"

# For IPython applications:
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename='LogOutput_1.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')


class Participant:
    # Dunder Methods
    def __init__(self, number, birthday, sex, height, weight, date_of_visit, age_months):
        """Initializes a new Participant instance with basic demographic and measurement data.
        
        Parameters:
        - number (str|int|float): Identifier for the participant.
        - birthday (str): Date of birth in format "MM/DD/YYYY".
        - sex (str): Gender of the participant.
        - height (float|str): Height in centimeters.
        - weight (float|str): Weight in kilograms.
        - date_of_visit (str): Date of the visit in format "MM/DD/YYYY".
        - age_months (float|str): Age in months. Calculated from birthday and date_of_visit if both values are present. Otherwise, accepts the provided value.

        Attributes:
        - BMI (float): Calculated BMI if height and weight are available, else None.
        """
        
        self.number = number
        self.birthday = self.try_parse_date(birthday)
        self.sex = self.try_assign_sex(sex)
        self.height = self.try_convert_float(height)
        self.weight = self.try_convert_float(weight)
        self.date_of_visit = self.try_parse_date(date_of_visit)
        self.age_months = self.set_age_months(age_months)
        
        if self.height and self.weight:
            self.BMI = self.weight/((self.height/100)**2)
        else:
            self.BMI = None 
        
        
    def __repr__(self):
        """Provides a developer-friendly string representation of the Participant instance, displaying all its attributes."""
        
        return str(self.__dict__)
    
    
    # Helper Methods for __init__ which check the validity of input data and raise errors accordingly. None values are returned as None.
    def try_parse_date(self, date_string: Optional[str]) -> Optional[datetime.date]:
        """
        Attempts to parse a string into a Date object using the MM/DD/YYYY format.

        Args:
        - date_string (str): The string representation of the date.

        Returns:
        - date (Date): Parsed date object if successful.

        Raises:
        - ValueError: If the string cannot be parsed into a valid date in the expected format.
        """

        if date_string is None:
            return None

        try:
            return datetime.strptime(date_string, "%m/%d/%Y").date()
        except ValueError:
            raise ValueError(f"Unable to understand {date_string} in 'birthday'/'date_of_vist' field. Please ensure dates are in MM/DD/YYYY format.")
      
    
    def try_assign_sex(self, input_string: Optional[str]) -> Optional[str]:
        """
        Determines the gender based on the provided input string.
        
        Args:
        - input_string (str): The input representing the gender. Accepted values include 'M', 'm', '1' for Male and 'F', 'f', '2' for Female.
        
        Returns:
        - str: 'M' for Male and 'F' for Female if the input is valid. None if the input is None.
        
        Raises:
        - ValueError: If the input_string does not match the expected values.
        """

        if input_string is None:
            return None

        male_vals = ['1', 'M', 'm']
        female_vals = ['2', 'F', 'f']

        if input_string in male_vals:
            return 'M'
        elif input_string in female_vals:
            return 'F'
        else:
            raise ValueError(f"Unable to understand {input_string} in 'sex' field. Accepted values: Male = 1/M/m; Female = 2/F/f. If unknown, leave blank.")
      
    
    def try_convert_float(self, value_string: Optional[str]) -> Optional[float]:
        """
        Converts a string to a float and ensures it's a positive value.
        
        Args:
        - value_string (str): The string representation of a number.
        
        Returns:
        - float_val (float): The converted float value if valid and positive.
        
        Raises:
        - ValueError: If the string cannot be converted to a positive float.
        """

        if value_string is None:
            return None
        
        try:
            float_val = float(value_string)
        except (ValueError, TypeError):
            raise ValueError(f"Unable to understand {value_string} in 'height'/'weight' field. Input must be a number. If unknown, leave blank.")
        
        if float_val <= 0:
            raise ValueError(f"Value {float_val} in 'height'/'weight' field must be positive.")
            
        return float_val
    
    
    def set_age_months(self, input_age_months: Optional[str]) -> Optional[int]:
        """
        Calculates the age of the participant in months based on the participant's date of birth and visit date.
        If data is unavailable, uses the provided age in months.
        
        Args:
        - input_age_months (str): The string representation of a number.
        
        Returns:
        - age_months (int): The age in months of the participant, rounded to the nearest whole number.
        
        Raises:
        - ValueError: If computed age is negative or if the provided age in months cannot be converted to a positive integer.
        """
        
        if self.birthday and self.date_of_visit:
            age_days = (self.date_of_visit - self.birthday).days
            if age_days < 0:
                raise ValueError(f"date_of_visit ({self.date_of_visit}) cannot be before birthday ({self.birthday}). If values are unknown, leave blank.")
            return int(round(age_days / DAYS_PER_MONTH))
        
        if input_age_months is None:
            return None
        
        try:
            age_months = int(round(float(input_age_months)))
        except ValueError:
            raise ValueError(f"Unable to understand {input_age_months} in 'age_months' field. Input must be a positive number. If unknown, leave blank or provide birthday and date_of_visit.")
        
        if age_months < 0:
            raise ValueError(f"Unable to understand {input_age_months} in 'age_months' field. Input must be a positive number. If unknown, leave blank or provide birthday and date_of_visit.")
            
        return age_months
        
        
    # Methods used for assignment of individual LMS values and z-scores
    def assign_LMS_values(self, LMS_table: List[dict]) -> None:
        """
        Assigns appropriate LMS values as instance variabbles to a participant based on their age in months.
        Optionally assigns a Standard Deviation (SD) if calculating participant's height-for-age z-score.
        
        Args:
        - LMS_table (List[dict]): A list of dictionaries each mapping keys ('L', 'M', 'S', and optionally 'SD') to their respective values.
          The index in the list corresponds to the age in months.
        """

        row = LMS_table[self.age_months]
        self.L = float(row['L'])
        self.M = float(row['M'])
        self.S = float(row['S'])
        self.SD = float(row.get('SD', 0))
    
        
    def calc_z_score_boxcox(self, indicator_type: Literal['wfa', 'bmifa']) -> None:
        """
        Applies Cole's adaptation of the Box-Cox transformation to calculate a z-score for either weight-for-age (wfa) or BMI-for-age (bmifa) and sets this z-score
        as an instance variable.
        
        Args:
        - indicator_type (str): Type of anthropometric indicator for which the z-score is to be calculated. Accepted values include 'wfa' and 'bmifa'.
        """
        
        if indicator_type=='wfa':
            z_string = 'wfa_z_score'
            x = self.weight
        else:
            z_string = 'bmifa_z_score'
            x = self.BMI
        
        L = self.L
        M = self.M
        S = self.S
        
        z = (((x/M)**L)-1)/(L*S)
        
        # Accounting for the lack of empirical data at extreme z-scores:
        if z > 3:
            SD3 = M*(1+(3*L*S))**(1/L)
            SD23 = SD3 - M*(1+(2*L*S))**(1/L)
            z = 3 + (x-SD3)/SD23
        
        if z < -3:
            SD3 = M*(1+(-3*L*S))**(1/L)
            SD23 = M*(1+(-2*L*S))**(1/L) - SD3
            z = -3 + (x-SD3)/SD23
            
        setattr(self, z_string, z)
    
    
    def calc_z_score_standard(self) -> None:
        """Calculates the z-score for height-for-age using the standard score formula and sets it as an instance variable."""

        x = self.height
        L = self.L
        M = self.M
        S = self.S
        SD = self.SD
        
        z = (x-M)/SD    
        
        self.hfa_z_score = z


def import_LMS_table(filename: str) -> List[dict]:
    """
    Imports a provided CSV file containing the LMS values for a given gender and indicator.
    
    Args:
    - filename (str): The name of the CSV file (including the .csv extension) which contains the LMS table.
    
    Returns:
    - LMS_table (List[dict]): A list of dictionaries each mapping keys ('L', 'M', 'S', and optionally 'SD') to their respective values.
      The index in the list corresponds to the age in months.
    
    Example:
    - LMS_table[0] contains the entry for age_months=0.
    """
    
    LMS_table = []
    with open(filename, newline='') as LMS_table_file:
        reader = csv.DictReader(LMS_table_file)
        for row in reader:
            LMS_table.append(row)
    
    return LMS_table
    
    
def import_all_LMS_tables(LMS_file_names: Dict[str, Tuple[str, str]]) -> dict:
    """
    Imports LMS tables for multiple indicator types from specified CSV files.

    Given a dictionary of indicator types (e.g., 'wfa', 'hfa', 'bmifa') mapped to tuples containing male and female CSV file names,
    this function reads the data and returns a nested dictionary structure. Each indicator type maps to another dictionary containing 
    the 'M' and 'F' keys, which represent the LMS data for males and females respectively.

    Parameters:
    - LMS_file_names (dict): A dictionary where the key is the indicator type and the value is a tuple containing the male and female CSV file names respectively.

    Returns:
    - all_LMS_tables (dict): A nested dictionary containing LMS data for each indicator type, separated by gender.

    Example:
        Input:
        LMS_file_names = {
            'wfa': ('wfa_boys_0_to_10_years_LMS.csv', 'wfa_girls_0_to_10_years_LMS.csv'),
            ...
        }

        Output:
        {
            'wfa': {
                'M': <data from wfa_boys_0_to_10_years_LMS.csv>,
                'F': <data from wfa_girls_0_to_10_years_LMS.csv>
            },
            ...
        }
    """
    
    all_LMS_tables = {}
    for indicator_type, (male_file_name, female_file_name) in LMS_file_names.items():
        all_LMS_tables[indicator_type] = {'M': import_LMS_table(male_file_name), 'F': import_LMS_table(female_file_name)}
    return all_LMS_tables
    
    
def import_participant_data(filename: str) -> List[Participant]:
    """
    Imports participant data from a provided CSV file and creates Participant objects.
    
    Args:
    - filename (str): The name of the CSV file (including the .csv extension) which contains the LMS table.
    
    Returns:
    - all_participant_objects: A list of all the newly created Participant objects.
    
    """
    
    all_participant_objects = []
    
    with open(filename, newline='') as participants_csv:
        reader = csv.DictReader(participants_csv)
        for participant in reader:
            new_participant = Participant(participant.get('number'), 
                          participant.get('birthday') or None,
                          participant.get('sex') or None,
                          participant.get('height') or None,
                          participant.get('weight') or None,
                          participant.get('date_of_visit') or None,
                          participant.get('age_months') or None
                         )
            all_participant_objects.append(new_participant)
    return all_participant_objects


def compute_participants(participants: List[Participant], LMS_dict: Dict[str, Dict[str, List[Dict[str, str]]]]):
    """
    Computes the z-scores for a list of participants based on the provided LMS values. Modifies the Participant objects directly.
    
    Args:
    - participants (List[Participant]): A list of the Participants objects for whom z-scores should be computed.
    - LMS_dict (Dict[str, Dict[str, List[Dict[str, str]]]]): A dictionary containing LMS tables for various indicators separated by sex.
    
    Notes:
    - This function directly modifies the Participant objects in the participants list.
    - Computation is only performed for participants below 19 years of age, with a valid sex and age.
    - Z-scores are computed for wfa, hfa, and bmifa based on the availability of weight and height data.
    - If either weight or height data is missing, corresponding z-scores are skipped and warnings are logged.
    """
    
    for participant in participants:
        age_months = participant.age_months
        sex = participant.sex
        height = participant.height
        weight = participant.weight
        number = participant.number
        
        if age_months is None or sex is None:
            logging.warning("Participant #%s: Participant missing age (%s) or sex (%s) or both. Skipping computation.", number, age_months, sex)
            continue
            
        if age_months > MAX_AGE_MONTHS_FOR_CALC:
            logging.warning("Participant #%s: Participant aged 19 or older (%s months). Skipping computation.", number, age_months)
            continue
        
        # Weight-for-Age Calculation
        if weight:
            if age_months < MAX_AGE_MONTHS_FOR_WFA:
                LMS_table = LMS_dict['wfa'][sex]
                participant.assign_LMS_values(LMS_table)
                participant.calc_z_score_boxcox('wfa')
            else:
                logging.warning("Participant #%s: Participant aged 10 or older (%s months). Unable to calculate wfa.", number, age_months)
        else:
            logging.warning("Participant #%s: Missing weight. Unable to calculate wfa and bmifa.", number)
        
        # Height-for-Age Calculation
        if height:
            LMS_table = LMS_dict['hfa'][sex]
            participant.assign_LMS_values(LMS_table)
            participant.calc_z_score_standard()       
        else:
            logging.warning("Participant #%s: Missing height. Unable to calculate hfa and bmifa.", number)
            
        # BMI-for-Age Calculation
        if height and weight:
            LMS_table = LMS_dict['bmifa'][sex]
            participant.assign_LMS_values(LMS_table)
            participant.calc_z_score_boxcox('bmifa')


def main():
    print("This program will calculate HFA, WFA, and BMIFA z-scores from raw data on participants aged 0-19 (WFA age 0-10).")
    print("Please save your data in a CSV file with column headers called 'number', 'birthday', 'sex', 'height', 'weight', 'date_of_visit', and 'age_months'. Order of columns and presence of other extraneous columns is unimportant.")
    print("All dates should be in MM/DD/YYYY format. 'age_months' may be omitted if both 'birthday' and 'date_of_visit' are present.")
    print("Weight should be in kilograms. Height should be in centimeters.")
    print("Missing data will result in skipped computation accordingly. 'number' is unnecessary and for subject tracking only.")
    input_participant_csv = input("Enter the name of the CSV file containing your data (including the .csv extension): ")

    all_LMS_tables_dict = import_all_LMS_tables(LMS_FILE_NAMES)
    all_participant_objects = import_participant_data(input_participant_csv)

    compute_participants(all_participant_objects, all_LMS_tables_dict)

    output_df = pd.DataFrame([participant.__dict__ for participant in all_participant_objects])
    output_df.drop(['L', 'M', 'S', 'SD'], axis=1, inplace=True)
    
    display(output_df)

    output_file_name = input(f"Enter the name you'd like the output csv file to have (including the .csv extension), or press enter for default {DEFAULT_OUTPUT_NAME}: ")
    if not output_file_name:
        output_file_name = DEFAULT_OUTPUT_NAME
    output_df.to_csv(output_file_name)
    

if __name__ == "__main__":
    main()

