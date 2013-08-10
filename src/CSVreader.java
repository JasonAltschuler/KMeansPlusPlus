/*************************************************************************
 * @author Jason Altschuler
 * 
 * PURPOSE: Read CSV files
 ************************************************************************/

import java.io.FileReader;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;

public class CSVreader {

   /**
    * Reads double[][] from csv file
    */
   public static double[][] read(String inFile, int rows, int columns) {
      if (rows <= 0 || columns <= 0)
         throw new IllegalArgumentException("Invalid dimensions");

      BufferedReader bf = null;
      double[][] arr = new double[rows][columns];

      String line = "";
      int r = 0;
      int c;

      try {
         bf = new BufferedReader(new FileReader(inFile));
         
         while ((line = bf.readLine()) != null) {
            String[] x = line.split(",");

            if (x.length != columns)
               throw new IllegalArgumentException("File has invalid dimensions (columns)");
         
            for (c = 0; c < columns; c++)
               arr[r][c] = Double.parseDouble(x[c] + "\t");
            
            r++;
         }
         
      } catch (FileNotFoundException e) {
         e.printStackTrace();
      } catch (IOException e) {
         e.printStackTrace();
      } finally {
         if (bf != null) {
            try {
               bf.close();
            } catch (IOException e) {
               e.printStackTrace();
            }
         }
      }

      if (r != rows)
         throw new IllegalArgumentException("File has invalid dimensions (rows)");

      return arr;
   }   

   public static void main(String[] args) {
      String testFile;
      int rows;
      int columns;
      boolean print = false;

      // read user's configurations for test
      if (args.length == 4) {
         testFile = args[0];
         rows = Integer.parseInt(args[1]);
         columns = Integer.parseInt(args[2]);
         print = Boolean.parseBoolean(args[3]);
      }

      // default test case
      else {
         testFile = "ImageForOtsus.csv";
         rows = 1040;
         columns = 1392;
         print = false;
      }

      double[][] test = CSVreader.read(testFile, rows, columns);
      
      if (print) {
         for (int i = 0; i < test.length; i++) {
            for (int j = 0; j < test[0].length; j++)
               System.out.print(test[i][j] + " ");
            System.out.println();
         }
      }
      
      System.out.println("Read " + testFile + " successfully!");
   }
}