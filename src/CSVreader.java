/*************************************************************************
 * @author  Altschuler and Wu Lab
 * 
 * Reads CSV files 
 * 
 * TODO: Remove stupid comments. Remove main method. read(..) doesn't
 * throw graceful exceptions -- is this a problem? See call hierarchy
 ************************************************************************/

import au.com.bytecode.opencsv.CSVReader;

import java.io.FileReader;
import java.io.IOException;


public class CSVreader {

    //"C:\\Users\\Steve\\Documents\\MATLAB\\test1.csv"
    // FOR K-MEANS, USE ROWS = 3000, COLUMNS = 2.
    /**
     * Reads double[][] from csv .txt file
     */
    public static double[][] read(String inFile, int rows, int columns) throws IOException {
        if (rows <= 0 || columns <= 0)
            throw new IllegalArgumentException("Invalid dimensions");
        
        CSVReader reader = new CSVReader(new FileReader(inFile));
        double[][] arr = new double[rows][columns];
        
		String[] nextLine;
		int r = 0;
		int c;
		
		while ((nextLine = reader.readNext()) != null) {
		    for (c = 0; c < columns; c++)
		        arr[r][c] = Double.parseDouble(nextLine[c]);
		        
		    r++;
		}

//		if (r != rows)
//	       throw new IllegalArgumentException("File has invalid dimensions (rows)");
		
        reader.close();

		return arr;
	}   
    
    public static void main(String[] args) {
        try {
            double[][] test = CSVreader.read("C:\\Users\\Jason\\Desktop\\Programming\\Matlab\\UT Southwestern work\\test1.csv", 200, 2);
       
           for (int i = 0; i < test.length; i++) {
               for (int j = 0; j < test[0].length; j++)
                   System.out.print(test[i][j] + " ");
               System.out.println();
           }
           
        } catch (IOException e) {
            e.printStackTrace();
        }
      
        System.out.println("completed");
    }
}