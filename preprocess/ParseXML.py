import pandas as pd 
import numpy as np 
import xml.etree.cElementTree as et


def getValueofNode(node):
    return node.text if node is not None else None

def XML2DF(fileName, columnNames):
    parsed_xml = et.parse(fileName);
    df_xml = pd.DataFrame(columns=columnNames)

    for node in parsed_xml.getroot():
        df_xml = df_xml.append(pd.Series([node.attrib.get(col) for col in columnNames], index=columnNames),ignore_index=True)
    
    return df_xml
