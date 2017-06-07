import java.util.*;
import java.io.*;

public class madaline {
    private static int inDimen, outDimen, numOfPairs, maxEpochs;
    private static int epochs = 0;
    private static int targets[], inputs[][], results[];
    private static double theta, alpha;
    private static double inWeights[][], biases[], hiddenWeights[] = {0.5, 0.5, 0.5};

    //main method receives user input for option selection, as well as necessary input for selected option
    public static void main(String[] args) {
        int random, response = 0;
        String weightFile, resultsFile, trainFile;
        Scanner input = new Scanner(System.in);
        while(true) {
            System.out.println("Enter the 1 to train, 2 to test/deploy, or 3 to quit the network:");
            response = input.nextInt();
            if (response == 2) {
                System.out.println("Enter name of file where testing/deploying results will be saved:");
                resultsFile = input.next();
                test(); //runs test function
                try {
                    writeOutputToFile(resultsFile); //writes results of test
                } catch (Exception e) {
                    e.printStackTrace(); //throws exception if results could not be wrote to file
                }
            } else if (response == 3) {
                System.exit(1); //quits program if 3 is entered
            } else { //gets parameters for train function
                System.out.println("Enter the data file name:");
                trainFile = input.next();
                readFile(trainFile);
                System.out.println("Enter 0 to initialize weights to 0 enter 1 to initialize weights to a random value between -0.5 and 0.5:");
                random = input.nextInt();
                initWeights(random);
                System.out.println("Enter the maximum number of training epochs:");
                maxEpochs = input.nextInt();
                System.out.println("Enter the desired learning rate:");
                alpha = input.nextDouble();
                System.out.println("Enter threshold:");
                theta = input.nextDouble();
                System.out.println("Enter a file name to save the trained weights:");
                weightFile = input.next();
                train(); //runs train funcion

                System.out.println("Training converged after " + epochs + " epochs");
                try {
                    writeWeightsToFile(weightFile); //writes discovered weights to file
                } catch (Exception e) {
                    e.printStackTrace(); //throws exception if write to file fails
                }
            }
        }

    }

    //uses train file to discover satisfactory weights
    public static void train() {
        boolean converged = false; //initializes bool that allows stopping of while statement once converge need is satisfied
        double maxChange = 0.0; // initializes int to contain max change, necessary to discover if converge need is meeting requirement
        double currChange = 0.0;
        int j, q, i, b;
        int inputXs[] = new int[inDimen];
        while(!converged && epochs < maxEpochs) { //runs until converge is reached or max epochs is reached
            maxChange = 0.0;
            for(j = 0; j < numOfPairs; j++) { //runs for all sample inputs (one epoch)
                double z1In = 0, z2In = 0, z1 = 0, z2 = 0, yIn = 0, y = 0; // initializes local variables to aide in discovery of output

                //gets inputs for current epoch
                for(i = 0; i < inDimen; i++) {
                    inputXs[i] = inputs[j][i]; 
                }
                //calculates z1 input and z2 input with current weights and biases
                for(q = 0; q < 2; q++) {
                    z1In = z1In + inputXs[q] * inWeights[q][0];
                }
                z1In = z1In + biases[0];
                for(b = 0; b < 2; b++) {
                    z2In = z2In + inputXs[b] * inWeights[b][1];
                }
                z2In = z2In + biases[1];

                //applies activation function
                if(z1In >= 0) {
                    z1 = 1;
                }
                else {
                    z1 = -1;
                }
                if(z2In >= 0) {
                    z2 = 1;
                }
                else {
                    z2 = -1;
                }

                //calculates yIn and applies activation function to find output value (y)
                yIn = z1*hiddenWeights[0] + z2*hiddenWeights[1] + hiddenWeights[2];
                if(yIn >= 0) {
                    y = 1;
                }
                else {
                    y = -1;
                }

                int target = targets[j]; //gets target output value
                if(target != y) { //runs weight altering code if target output and achieved output are different
                    //if target is -1 then the affecting weights for all positive hidden neurons are changed, maxChange is updated to reflect highest amount of change a weight has undergone
                    if(target == -1) {
                        if(z1 > 0) {
                            for(int c = 0; c < 2; c++) {
                                inWeights[c][0] = inWeights[c][0] + alpha*(target - z1In)*inputXs[c];
                                currChange = Math.abs(alpha*(target - z1In)*inputXs[c]);
                                if(currChange > maxChange) {
                                    maxChange = currChange;
                                }
                            }
                            biases[0] = biases[0] + alpha*(target - z1In);
                            currChange = Math.abs(alpha*(target - z1In));
                            if(currChange > maxChange) {
                                maxChange = currChange;
                            }
                        }

                        if(z2 > 0) {
                            for(int g = 0; g < 2; g++) {
                                inWeights[g][1] = inWeights[g][1] + alpha*(target - z2In)*inputXs[g];
                                currChange = Math.abs(alpha*(target - z2In)*inputXs[g]);
                                if(currChange > maxChange) {
                                    maxChange = currChange;
                                }
                            }
                            biases[1] = biases[1] + alpha*(target - z2In);
                            currChange = Math.abs(alpha*(target - z2In));
                            if(currChange > maxChange) {
                                maxChange = currChange;
                            }
                        }
                    }

                    //if target is equal to 1 then the weights associated with the zIn closes to zero are changed,  maxChange is updated to reflect highest amount of change a weight has undergone
                    if(target == 1) {
                        double z1InAbs = Math.abs(z1In);
                        double z2InAbs = Math.abs(z2In);
                        if(z1InAbs < z2InAbs) {
                            for(int c = 0; c < 2; c++) {
                                inWeights[c][0] = inWeights[c][0] + alpha*(target - z1In)*inputXs[c];
                                currChange = Math.abs(alpha*(target - z1In)*inputXs[c]);

                                if(currChange > maxChange) {
                                    maxChange = currChange;
                                }
                            }
                            biases[0] = biases[0] + alpha*(target - z1In);
                            currChange = Math.abs(alpha*(target - z1In));
                            if(currChange > maxChange) {
                                maxChange = currChange;
                            }
                        }
                        else {
                            for(int g = 0; g < 2; g++) {
                                inWeights[g][1] = inWeights[g][1] + alpha*(target - z2In)*inputXs[g];
                                currChange = Math.abs(alpha*(target - z2In)*inputXs[g]);
                                if(currChange > maxChange) {
                                    maxChange = currChange;
                                }
                            }
                            biases[1] = biases[1] + alpha*(target - z2In);
                            currChange = Math.abs(alpha*(target - z2In));
                            if(currChange > maxChange) {
                                maxChange = currChange;
                            }
                        }
                    }

                }

            }
            if(maxChange < theta) { //if the max change a weight has undergone is less than the acceptable amount then the training is converged and the while loop ends
                converged = true;
            }
            epochs++; //epochs is increased by one to indicate the completion of a round of achieved output and desired output checking/weight altering
        }
    }

    //following method tests the discovered weights
    public static void test() {
        int k;
        for(k = 0;k < numOfPairs; k++) { //runs for all inputs provided
            double z1In = 0, z2In = 0, yIn = 0;
            int z1 = 0, z2 = 0;
            int y = 0;

            //calculates z1 and z2 then applies activation function to get inputs to be used by hidden neurons
            z1In = inWeights[0][0] * inputs[k][0] + inWeights[1][0] * inputs[k][1] + biases[0];
            z2In = inWeights[0][1] * inputs[k][0] + inWeights[1][1] * inputs[k][1] + biases[1];
            if(z1In >= 0) {
                z1 = 1;
            }
            else {
                z1 = -1;
            }
            if(z2In >= 0) {
                z2 = 1;
            }
            else {
                z2 = -1;
            }

            //yIn is calculated and passed through activation function in order to find final output of current input test
            yIn = z1 * hiddenWeights[0] + z2 * hiddenWeights[1] + hiddenWeights[2]; 
            if(yIn >= 0) {
                    y = 1;
           }
           else {
                    y = -1;
           }
           results[k] = y; //output of current input test is saved to results
        }
    }

    //following method writes found results to file with specified name
    public static void writeOutputToFile(String filename) {
        try {
            File testOutFile = new File(filename);
            FileWriter fw = new FileWriter(testOutFile.getAbsoluteFile());
            BufferedWriter bw = new BufferedWriter(fw);
            for (int j = 0; j < numOfPairs; j++) {
                bw.write(Double.toString(results[j]) + " ");
                bw.write("\n");
            }
            bw.close();
        }
        catch(Exception e){
            e.printStackTrace();
        }
    }

    //following method writes discovered weights to file with specified name
    public static void writeWeightsToFile(String filename) {
        try {
            File outputFile = new File(filename);
            FileWriter fw = new FileWriter(outputFile.getAbsoluteFile());
            BufferedWriter bw = new BufferedWriter(fw);
            for(int i = 0; i < 2; i++) {
                for(int j = 0; j < 2; j++) {
                    bw.write(Double.toString(inWeights[i][j]) + " ");
                }
                bw.write(biases[i] + " ");
            }
            bw.close();
        }
        catch(Exception e) {
            e.printStackTrace();
        }
    }

    //following method initializes weights to user specified amounts
    public static void initWeights(int ranOrNot) {
        if(ranOrNot == 0){
            for(int j = 0; j < 2; j++) {
                for(int i = 0; i < 2; i++) {
                    inWeights[j][i] = 0;
                }
                biases[j] = 0;
            }
        }
        else {
            for(int j = 0; j < 2; j++) {
                for(int i = 0; i < 2; i++) {
                    inWeights[j][i] = Math.random() - 0.5;
                }
                biases[j] = Math.random() - 0.5;
            }
        }
    }

    //following method reads file, extracting from it the dimensions of input and output, number of input pairs, actual inputs, and desired outputs for provided inputs
    public static void readFile(String filename) {
        try {
            Scanner scanner = new Scanner(new File(filename));
            inDimen = scanner.nextInt();
            outDimen = scanner.nextInt();
            numOfPairs = scanner.nextInt();
            inWeights = new double[inDimen][inDimen];
            biases = new double[inDimen];
            targets = new int[numOfPairs];
            inputs = new int[numOfPairs][inDimen];
            results = new int[numOfPairs];
            for(int t = 0; t < numOfPairs; t++) {
                for(int i = 0; i < inDimen; i++) {
                    inputs[t][i] = scanner.nextInt();
                }
                targets[t] = scanner.nextInt();
            }
        }
        catch(FileNotFoundException e){
            e.printStackTrace();
            System.exit(1);
        }
    }
}
