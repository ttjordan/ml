package neu.edu.ml.homework1;

import java.io.File;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileNotFoundException;
import java.io.IOException;

/**
 * Created by Tadeusz Jordan.
 */
public class Parser {

    public Parser() {
    }

    public HousingDataRows parseFile(String fileName) {

        HousingDataRows housingData = new HousingDataRows();

        File file = new File (fileName);
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {

            String line;
            int count = 0;

            while ( (line = br.readLine()) != null) {
                String[] tokens = line.trim().split("\\s+");

                HousingDataRow housingDataRow = new HousingDataRow(tokens);

                housingData.add(housingDataRow);

                count++;
                /*if(count == 4) {
                    return housingData;
                } */
            }

            System.out.println("Number of lines: " + count);

        } catch (FileNotFoundException e) {
            System.out.println("File not found");
        } catch (IOException e) {
            System.out.println("Unable to read file");
        }

        return housingData;
    }
}
