package kottmann.opennlp.addons.mallet;

import cc.mallet.types.FeatureConjunction;
import opennlp.tools.dictionary.Dictionary;
import opennlp.tools.ml.model.SequenceClassificationModel;
import opennlp.tools.postag.*;
import opennlp.tools.util.*;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) throws IOException {
        train();
        //predict();
    }

    static void train() throws IOException {
        InputStreamFactory isf = new MarkableFileInputStreamFactory(new File("demofile.txt"));
        ObjectStream<String> lineStream = new PlainTextByLineStream(isf, "UTF-8");
        ObjectStream<POSSample> sampleStream = new WordTagSampleStream(lineStream);
        POSSampleSequenceStream pss = new POSSampleSequenceStream(sampleStream);
        CRFTrainer trainer = new CRFTrainer();
        SequenceClassificationModel<String> model = trainer.train(pss);
        OutputStream modelOut = null;
        try {
            modelOut = new BufferedOutputStream(new FileOutputStream("trained.bin"));
            TransducerModelSerializer s = new TransducerModelSerializer();
            s.serialize((TransducerModel) model, modelOut);
        } catch (IOException e) {
            // Failed to save model
            e.printStackTrace();
        } finally {
            if (modelOut != null) {
                try {
                    modelOut.close();
                } catch (IOException e) {
                    // Failed to correctly save model.
                    // Written model might be invalid.
                    e.printStackTrace();
                }
            }
        }
    }

    static void predict() throws IOException {
        TransducerModelSerializer s = new TransducerModelSerializer();
        TransducerModel<String> model = s.create(new FileInputStream(new File("trained.bin")));
        InputStreamFactory isf2 = new MarkableFileInputStreamFactory(new File("demofile-test.txt"));
        ObjectStream<String> lineStream2 = new PlainTextByLineStream(isf2, "UTF-8");
        ObjectStream<POSSample> sampleStream2 = new WordTagSampleStream(lineStream2);
        ArrayList<POSSample> ar = new ArrayList<>();
        POSSample next = null;
        while((next = sampleStream2.read()) != null) {
            ar.add(next);
        }

        lineStream2 = new PlainTextByLineStream(isf2, "UTF-8");
        sampleStream2 = new WordTagSampleStream(lineStream2);
        //MutableTagDictionary dict = new POSDictionary();
        //POSTaggerME.populatePOSDictionary(sampleStream2, dict, 0);
        Dictionary dict = POSTaggerME.buildNGramDictionary(sampleStream2, 0);
        Measure m = new Measure();
        for(POSSample sam : ar) {
            Sequence seq = model.bestSequence(sam.getSentence(), sam.getAddictionalContext(), new DefaultPOSContextGenerator(dict), null);
            m.update(seq.getOutcomes().toArray(), sam.getTags());
            //seq.getOutcomes().forEach(System.out::print);
            //System.out.println();
        }
        System.out.println(m.getF1());
        System.out.println(m.getPrecision());
        System.out.println(m.getRecall());
    }
}
