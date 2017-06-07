import java.util.*;
import java.io.*;

public class perceptron  {

  private static int inDimen, outDimen, numOfPairs, maxEpochs;
  private static int epochs = 0, superConvCount = 0;
  private static int target[][], inputs[][], results[][], inputsTest[][], targetTest[][];
  private static double theta, alpha;
  private static double weights[][], biases[];
  private static char letters[];
  private static boolean superConv = false;

  public static void main(String args[]) {

    int random, response;
    String trainFile, weightFile, testFile, resultsFile, weightsFileIn;
    Scanner input = new Scanner(System.in);
    System.out.println("Enter the 1 to train using a training data file, enter 2 to train using a trained weights file:");
    response = input.nextInt();

    if(response == 2) {
          System.out.println("Enter the training data file name:");
          trainFile = input.next();
          readFile(trainFile);
          System.out.println("Enter trained weights file name:");
          weightsFileIn = input.next();
          getWeights(weightsFileIn);

          System.out.println("Enter the maximum number of training epochs:");
          maxEpochs = input.nextInt();
          System.out.println("Enter a file name to save the trained weights settings:");
          weightFile = input.next();
          System.out.println("Enter the learning rate (alpha) from 0 to 1 but not including 0:");
          alpha = input.nextDouble();
          System.out.println("Enter the threshold (theta):");
          theta = input.nextDouble();
          train();

          System.out.println("Training converged after " + epochs + " epochs");
          try {
              writeWeightsToFile(weightFile);
          }
          catch(Exception e) {
              e.printStackTrace();
          }
    }
    else {
       System.out.println("Enter the training data file name:");
       trainFile = input.next();
       readFile(trainFile);
       System.out.println("Enter 0 to initialize weights to 0 enter 1 to initialize weights to a small random value:");
       random = input.nextInt();
       initWeights(random);
       
       System.out.println("Enter the maximum number of training epochs:");
       maxEpochs = input.nextInt();
       System.out.println("Enter a file name to save the trained weights settings:");
       weightFile = input.next();
       System.out.println("Enter the learning rate (alpha) from 0 to 1 but not including 0:");
       alpha = input.nextDouble();
       System.out.println("Enter the threshold (theta):");
       theta = input.nextDouble();
       train();

       System.out.println("Training converged after " + epochs + " epochs");
       try {
              writeWeightsToFile(weightFile);
       }
       catch(Exception e) {
          e.printStackTrace();
       }
    }

    System.out.println("Enter 1 to test/deploy using a testing/deploying data file, enter 2 to quit:");
    response = input.nextInt();
    if(response == 1) {
       System.out.println("Enter the testing/deploying file name:");
       testFile = input.next();
       System.out.println("Enter a file name to save the testing/deploying results:");
       resultsFile = input.next();
       readTestFile(testFile);
       test();
          writeOutputToFile(resultsFile);
    }

  }

  public static void train() {

      boolean converged = false;

      double yIn = 0;
      double y = 0;
      int j, q, i, b;
      while(!superConv && epochs < maxEpochs) {
          converged = true;
          for(j = 0; j < numOfPairs; j++) {
              for(q = 0; q < outDimen; q++) {
                  for(i = 0; i < inDimen; i++) {
                      yIn += (weights[q][i] * inputs[j][i]);
                  }
                  yIn += biases[q];

                  for(b = 0; b < inDimen; b++) {

                      if(yIn > theta) {
                          y = 1;
                      }
                      else if(yIn < theta) {
                          y = -1;
                      }
                      else {
                          y = 0;
                      }

                      if(y != target[j][q]) {
                          converged = false;

                          yIn = yIn - (weights[q][b] * inputs[q][b]) - biases[q];

                          weights[q][b] += (alpha * target[j][q] * inputs[j][b]);

                          biases[q] += (alpha * target[j][q]);

                          yIn = yIn + (weights[q][b] * inputs[q][b]) + biases[q];
                      }
                  }
              }
          }
          if(converged == true) {
              superConvCount++;
              if(superConvCount == 15) {
                  superConv = true;
              }
          }
          else {
              superConvCount = 0;
          }

          epochs++;
      }
  }

  public static void test() {

      int yIn = 0;
      int y = 0;
      int k, i, j;
      for(k = 0;k < numOfPairs; k++) {
          for(i = 0; i < outDimen; i++) {

              for(j = 0; j < inDimen; j++) {
                  yIn += (weights[i][j] * inputsTest[k][j]);
              }

              yIn += biases[i];

              if(yIn > theta) {
                  y = 1;
              }
              else if(yIn < theta){
                  y = -1;
              }
              else{
                  y = 0;
              }

              results[k][i] = y;
          }
      }
  }

  public static void getWeights(String filename) {

      try {
          Scanner scanner = new Scanner(new File(filename));
          for(int j = 0; j < outDimen; j++) {
              for(int i = 0; i < inDimen; i++) {
                  weights[j][i] = scanner.nextDouble();
              }
              biases[j] = scanner.nextDouble();
          }
      }
      catch(FileNotFoundException e) {
          e.printStackTrace();
          System.exit(1);
      }

  }

  public static void writeOutputToFile(String filename) {

    try {

          File testOutFile = new File(filename);
          FileWriter fw = new FileWriter(testOutFile.getAbsoluteFile());
          BufferedWriter bw = new BufferedWriter(fw);
          for (int j = 0; j < numOfPairs; j++) {
          for (int i = 0; i < outDimen; i++) {
                  bw.write(Double.toString(results[j][i]) + " ");
          }

              bw.write("\n");

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
          for(int i = 0; i < outDimen; i++) {
          for(int j = 0; j < inDimen; j++) {
                  bw.write(Double.toString(weights[i][j]) + " ");
          }
              bw.write(biases[i] + " ");
       }

          bw.close();
    }
    catch(Exception e) {
       e.printStackTrace();
    }

  }

   public static void initWeights(int ranOrNot) {

      if(ranOrNot == 0){
         for(int j = 0; j < outDimen; j++) {
            for(int i = 0; i < inDimen; i++) {
               weights[j][i] = 0;
            }
            biases[j] = 0;
         }
      }
      else {
         for(int j = 0; j < outDimen; j++) {
            for(int i = 0; i < inDimen; i++) {
               weights[j][i] = smallRandom();
            }
            biases[j] = smallRandom();
         }
      }
   }

  public static double smallRandom() {

      return (Math.random() - .5) * .75;
  }

  public static void readFile(String filename) {

      String emptyLine;
      char [] letter = new char [2];

      try {

          Scanner scanner = new Scanner(new File(filename));
          inDimen = scanner.nextInt();
          outDimen = scanner.nextInt();
          numOfPairs = scanner.nextInt();
          weights = new double[outDimen][inDimen];
          biases = new double[outDimen];
          target = new int[numOfPairs][outDimen];
          inputs = new int[numOfPairs][inDimen];
          letters = new char[outDimen];

          int q = 0;
          for(int t = 0; t < numOfPairs; t++) {
              for(int i = 0; i < inDimen; i++) {
                  inputs[t][i] = scanner.nextInt();
              }

              for(int j = 0; j < outDimen; j++) {
                  target[t][j] = scanner.nextInt();
                  if(target[t][j] == 1) {
                      q = j;
                  }
              }
              letter = scanner.next().toCharArray();
              letters[q] = letter[0];
          }
      }
      catch(FileNotFoundException e){
          e.printStackTrace();
          System.exit(1);
      }
  }

  public static void readTestFile(String filename) {

    int t, i, j, q = 0;
    char[] letterAr = new char [2];
    try {
       Scanner scanner = new Scanner(new File(filename));
       inDimen = scanner.nextInt();
       outDimen = scanner.nextInt();
       numOfPairs = scanner.nextInt();

       inputsTest = new int[numOfPairs][inDimen];
       targetTest = new int[numOfPairs][outDimen];
       results = new int[numOfPairs][outDimen];
       for(t = 0; t < numOfPairs; t++){
          for(i = 0; i < inDimen + outDimen; i++) {
             if(i < inDimen) {
                inputsTest[t][i] = scanner.nextInt();
             }
             if(i >= inDimen) {
                targetTest[t][i - inDimen] = scanner.nextInt();
                if(targetTest[t][i - inDimen] == 1) {
                          q = i - inDimen;
                }
             }
           }
              letterAr = scanner.next().toCharArray();
           letters[q] = letterAr[0];
       }
    }  

    catch(FileNotFoundException e){
       e.printStackTrace();
       System.exit(1);
    }
  }

}  
