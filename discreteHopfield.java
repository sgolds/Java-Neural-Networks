import java.util.*;
import java.io.*;

public class discreteHopfield {

    private static int dimen, numOfVectors;
    private static double threshold = 0;
    private static int savedPatterns[][], testPatterns[][], results[][];
    private static double inWeights[][];
    private static ArrayList<String> stringsForResultsOutput = new ArrayList<String>();

    public static void main(String[] args) {

        //The following code collects the user input required for training/testing, e.g. threshold value and file names
        int response = 0;
        String weightFile, resultsFile, trainFile, testFile;
        Scanner input = new Scanner(System.in);
        while(true) {
            System.out.println("Enter the 1 to train, 2 to test/deploy, or 3 to quit the network:");
            response = input.nextInt();

            if (response == 2) {
                System.out.println("Enter the testing data file name:");
                testFile = input.next();
                System.out.println("Enter name of weight file to get weights from:");
                weightFile = input.next();
                System.out.println("Enter name of file where testing/deploying results will be saved:");
                resultsFile = input.next();

                readTestFile(testFile);
                readWeightFile(weightFile);
                test();
                try {
                    writeOutputToFile(resultsFile);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            } else if (response == 3) {
                System.exit(1);
            } else {
                System.out.println("Enter the training data file name:");
                trainFile = input.next();
                System.out.println("Enter threshold:");
                threshold = input.nextDouble();
                System.out.println("Enter a file name to save the trained weights:");
                weightFile = input.next();

                readFile(trainFile);
                createWeights();
                train();

                try {
                    writeWeightsToFile(weightFile);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }

    }

    public static void train() {
        //creates array list for use in random order yIn training
        ArrayList<Integer> randomOrder = new ArrayList<Integer>();
        Random r = new Random();
        int rand = 0;


        for(int i = 0; i < numOfVectors; i++) { //interates for all input vectors
            int[] y = new int[dimen];
            boolean converged = false;
            while (!converged) { //runs for as long as convergence takes
                converged = true;
                while (randomOrder.size() < 100) { //creates and fills random order training yIn list
                    rand = r.nextInt(dimen);
                    if (!(randomOrder.contains(rand))) {
                        randomOrder.add(rand);
                    }
                }
                y = savedPatterns[i]; //sets y to input vector
                for (Integer currNum : randomOrder) { //runs yIn broadcasting for all Ys in random order
                    double yIn = (double) savedPatterns[i][currNum]; //creates yIn with associated x value
                    for (int c = 0; c < dimen; c++) { //adds associated y * weight values to yIn
                        yIn = yIn + y[c] * inWeights[c][currNum];
                    }

                    //runs activation function on yIn
                    if (yIn > threshold) {
                        if(1 != y[currNum]) {
                            y[currNum] = 1; //broadcasts new y value
                            converged = false; //marks that a value was changed and convergence is not possible
                        }
                    } else if (yIn < threshold) {
                        if(-1 != y[currNum]) {
                            y[currNum] = -1; //broadcasts new y value
                            converged = false;
                        }
                    }
                }

                randomOrder.clear(); //clears random order to allow a new different random order
            }
        }

    }

    public static void test() {
        //for following code see train method
        ArrayList<Integer> randomOrder = new ArrayList<Integer>();
        Random r = new Random();
        int rand = 0;

        int[] y = new int[dimen];
        for(int i = 0; i < numOfVectors; i++) {
            boolean converged = false;
            while (!converged) {
                converged = true;
                while (randomOrder.size() < 100) {
                    rand = r.nextInt(dimen);
                    if (!(randomOrder.contains(rand))) {
                        randomOrder.add(rand);
                    }
                }
                y = testPatterns[i];
                for (Integer currNum : randomOrder) {
                    double yIn = (double) testPatterns[i][currNum];
                    for (int c = 0; c < dimen; c++) {
                        yIn = yIn + y[c] * inWeights[c][currNum];
                    }

                    if (yIn > threshold) {
                        if(1 != y[currNum]) {
                            y[currNum] = 1;
                            converged = false;
                        }
                    } else if (yIn < threshold) {
                        if(-1 != y[currNum]) {
                            y[currNum] = -1;
                            converged = false;
                        }
                    }
                }

                randomOrder.clear();
            }

            results[i] = y; //sets results array value to achieved result value

            //following code prepares for the writing to results file by creating strings to be written
            stringsForResultsOutput.add("Results for test input vector " + (i+1) +":\n");
            stringsForResultsOutput.add(Arrays.toString(y) + "\n");

            //checks if the achieved result array matched any saved pattern
            Boolean matched = false;
            for(int h = 0; h < numOfVectors; h++) {
                if(Arrays.equals(y, savedPatterns[h])) {
                    stringsForResultsOutput.add("Test input vector: " + (i + 1) + " Matched with saved pattern: " + (h + 1) + "\n\n");
                    matched = true;
                }
            }

            if(!matched) {
                stringsForResultsOutput.add("Test input vector did not match with any saved pattern\n\n");
            }
        }
    }

    public static void writeOutputToFile(String filename) {

        try {

            File testOutFile = new File(filename);
            FileWriter fw = new FileWriter(testOutFile.getAbsoluteFile());
            BufferedWriter bw = new BufferedWriter(fw);
            for(String s: stringsForResultsOutput) { //writes prepared strings to output in user friendly way
                bw.write(s);
            }

            bw.close();
        }
        catch(Exception e){
            e.printStackTrace();
        }
    }

    public static void writeWeightsToFile(String filename) {

        try {

            File outputFile = new File(filename);
            FileWriter fw = new FileWriter(outputFile.getAbsoluteFile());
            BufferedWriter bw = new BufferedWriter(fw);
            for(int i = 0; i < dimen; i++) {
                for(int j = 0; j < dimen; j++) {
                    bw.write(Double.toString(inWeights[i][j]) + " "); //writes weights with space between weights for easy reading
                }
                bw.write("\n"); //makes weight output appropriately 2d
            }

            bw.close();
        }
        catch(Exception e) {
            e.printStackTrace();
        }

    }

    public static void createWeights() {

        //creates weights using stored patterns, vector addition, vector multiplication, and vector transposing
        int i = 0, j = 0, c = 0;
        for(i = 0; i < dimen; i++) {
            for(j = 0; j < dimen; j++) {
                inWeights[i][j] = 0;
            }
        }
        for(i = 0; i < numOfVectors; i++) {
            for(j = 0; j < dimen; j++) {
                for(c = 0; c < dimen; c++) {
                    inWeights[j][c] = inWeights[j][c] + savedPatterns[i][j]*savedPatterns[i][c];
                }
            }
        }
        for(i = 0; i < dimen; i++) {
            inWeights[i][i] = 0; //eliminates self-connection
        }
    }

    public static void readFile(String filename) {

        try {

            Scanner scanner = new Scanner(new File(filename));

            String emptyLine = "";
            dimen = scanner.nextInt();
            numOfVectors = scanner.nextInt();
            scanner.useDelimiter("");
            savedPatterns = new int[numOfVectors][dimen];
            inWeights = new double[dimen][dimen];

            emptyLine = scanner.next();
            String next = "";
            for(int t = 0; t < numOfVectors; t++) { //gets all input vector patterns
                emptyLine = scanner.nextLine();
                for(int i = 0; i < dimen; i++) {
                    if(scanner.hasNext()) {
                        next = scanner.next();
                        if(!(next.equals("\n"))) {
                            //creates bipolar saved patterns
                            if(next.equals(" ")) {
                                savedPatterns[t][i] = -1; //sets spaces to -1
                            }
                            else if(next.equals("0") || next.equals("o") || next.equals("O")) {
                                savedPatterns[t][i] = 1;
                            }
                        }
                        else {
                            i = i -1;
                        }
                    }
                }
            }
            scanner.close();

        }
        catch(FileNotFoundException e){
            e.printStackTrace();
            System.exit(1);
        }
    }

    public static void readWeightFile(String filename) {

        try {

            Scanner scanner = new Scanner(new File(filename));

            //reads weights, accounting for possibility of doubles and ints
            for(int h = 0; h < dimen; h++) {
                for(int j = 0; j < dimen; j++) {
                    if(scanner.hasNextInt()) {
                        inWeights[h][j] = (double)scanner.nextInt();
                    }
                    else if(scanner.hasNextDouble()) {
                        inWeights[h][j] = scanner.nextDouble();
                    }
                }
            }

            scanner.close();

        }
        catch(FileNotFoundException e){
            e.printStackTrace();
            System.exit(1);
        }
    }

    public static void readTestFile(String filename) {
        //see readFile method
        try {

            Scanner scanner = new Scanner(new File(filename));

            String emptyLine = "";
            dimen = scanner.nextInt();
            numOfVectors = scanner.nextInt();
            scanner.useDelimiter("");
            testPatterns = new int[numOfVectors][dimen];
            results = new int[numOfVectors][dimen];

            emptyLine = scanner.next();
            String next = "";
            for(int t = 0; t < numOfVectors; t++) {
                emptyLine = scanner.next();
                for(int i = 0; i < dimen; i++) {
                    if(scanner.hasNext()) {
                        next = scanner.next();
                        if(!(next.equals("\n"))) {
                            if(next.equals(" ")) {
                                testPatterns[t][i] = -1;
                            }
                            else {
                                testPatterns[t][i] = 1;
                            }
                        }
                        else {
                            i = i -1;
                        }
                    }
                }
            }
            scanner.close();

        }
        catch(FileNotFoundException e){
            e.printStackTrace();
            System.exit(1);
        }
    }
}
