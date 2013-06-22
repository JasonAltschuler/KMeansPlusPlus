/*************************************************************************
 * @author Altschuler and Wu Lab
 * 
 * Writes CSV files 
 * 
 * TODO: Remove main method.
 ************************************************************************/

import java.io.PrintStream;
import java.io.FileOutputStream;
import java.io.FileNotFoundException;


public class CSVwriter {

   /**
    * Writes double[][] to .txt file
    */
   public static void write(String outFile, double[][] arr) throws FileNotFoundException {
      PrintStream out = new PrintStream(new FileOutputStream(outFile));

      int rows = arr.length;
      int columns = arr[0].length;

      for (int r = 0; r < rows; r++) {
         for (int c = 0; c < columns; c++) {
            out.print(arr[r][c]);

            if (c != columns - 1)
               out.print(',');
         }
         out.println();
      }

      out.close();
   }

   
   /**
    * Unit testing
    */
   public static void main(String[] args) throws FileNotFoundException {
      double[][] test = {{1, 2, 3}, 
            {4, 5, 6}};

      CSVwriter.write("testWrite", test);        
   }

}