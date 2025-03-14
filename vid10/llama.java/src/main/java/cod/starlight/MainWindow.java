package cod.starlight;

import javax.swing.*;
import javax.swing.text.BadLocationException;
import javax.swing.text.SimpleAttributeSet;
import javax.swing.text.StyleConstants;
import javax.swing.text.StyledDocument;
import java.awt.*;
import java.awt.event.ComponentEvent;
import java.awt.event.ComponentListener;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public class MainWindow extends JFrame {
    private JTabbedPane tabbedPane;
    private JTextArea inputTextArea;
    private JTextPane readOnlyTextArea;
    private JSplitPane splitPane;
    private JScrollPane readOnlyScrollPane;

    public MainWindow() {
        this.setTitle("Simple Window");
        this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        this.setSize(700, 600);

        // Create the read-only text area
        readOnlyTextArea = new JTextPane();
        readOnlyTextArea.setContentType("text/html");
        readOnlyTextArea.setEditable(false);

        // Create the input text area
        inputTextArea = new JTextArea();
        Font f = inputTextArea.getFont();
        Font nf = new Font(f.getFontName(), f.getStyle(), (int) (f.getSize() * 1.5));
        inputTextArea.setFont(nf);
        inputTextArea.setText("Why the sky is blue?");
        inputTextArea.setCaretPosition(inputTextArea.getText().length());
        readOnlyScrollPane = new JScrollPane(readOnlyTextArea);
        JScrollPane inputScrollPane = new JScrollPane(inputTextArea);

        splitPane = new JSplitPane(JSplitPane.VERTICAL_SPLIT, readOnlyScrollPane, inputScrollPane);
        splitPane.setDividerLocation(470); // Set the initial position of the divider

        // Add the panel to the frame
        this.add(splitPane);

        this.addComponentListener(new ComponentListener() {
            @Override
            public void componentResized(ComponentEvent e) {

            }

            @Override
            public void componentMoved(ComponentEvent e) {

            }

            @Override
            public void componentShown(ComponentEvent e) {
                inputTextArea.grabFocus();
                try {
                    initChat();
                } catch (Exception ex) {
                    addError(ex);
                }

                inputTextArea.addKeyListener(new KeyAdapter() {
                    @Override
                    public void keyPressed(KeyEvent e) {
                        if (e.getKeyCode() == KeyEvent.VK_ENTER) {
                            String txt = inputTextArea.getText();
                            if (txt != null) {
                                if (!txt.isEmpty()) {
                                    try {
                                        chat(inputTextArea.getText());
                                    } catch (Exception ex) {
                                        addError(ex);
                                    }
                                    inputTextArea.setText("");
                                }
                            }
                        }
                    }
                });
            }

            @Override
            public void componentHidden(ComponentEvent e) {

            }
        });


        // Make the frame visible
        this.setVisible(true);
    }

    private void addError(Exception ex) {
        addError(ex.getMessage());
    }

    private void addError(String str) {
        addText(str, Color.RED, true, true);
    }

    private void addInfo(String str) {
        addText(str, Color.GREEN, false, true);
    }

    private void printChat(String str) {
        addText(str, Color.blue, false, false);
    }

    private void printUserQuestion(String str) {
        addText(str, Color.black, true, true);
        addText("------------------------------\n", Color.black, true, false);
    }

    private void addText(String txt, Color color, boolean bold, boolean italic) {
        StyledDocument doc = readOnlyTextArea.getStyledDocument();

        // Add some styles to the document
        SimpleAttributeSet style = new SimpleAttributeSet();
        StyleConstants.setBold(style, bold);
        StyleConstants.setItalic(style, italic);
        StyleConstants.setForeground(style, color);
        StyleConstants.setFontSize(style, 14);

        try {
            doc.insertString(doc.getLength(), txt, style);
            int max = readOnlyScrollPane.getVerticalScrollBar().getMaximum();
            readOnlyScrollPane.getVerticalScrollBar().setValue(max);
        } catch (BadLocationException e) {
            throw new RuntimeException(e);
        }
    }


    //static final int DEFAULT_MAX_TOKENS = 512;
    private Llama model = null;
    private Sampler sampler;
    private Llama3.Options options;


    private Llama.State state = null;
    private java.util.List<Integer> conversationTokens = null;
    private ChatFormat chatFormat = null;
    private int startPosition = 0;


    private void initChat() throws IOException {
        String modelFile = "--model=D:\\LlmIs\\repo\\models\\Llama-3.2-1B-Instruct-DOA_q8_0.gguf";
        options = Llama3.Options.parseOptions(new String[]{"-i", modelFile,
                "--max-tokens=13000"});

        model = AOT.tryUsePreLoaded(options.modelPath(), options.maxTokens());
        if (model == null) {
            // No compatible preloaded model found, fallback to fully parse and load the specified file.
            model = ModelLoader.loadModel(options.modelPath(), options.maxTokens(), true);
        }
        sampler = Llama3.selectSampler(model.configuration().vocabularySize, options.temperature(), options.topp(), options.seed());


        conversationTokens = new ArrayList<>();
        chatFormat = new ChatFormat(model.tokenizer());
        conversationTokens.add(chatFormat.beginOfText);
        if (options.systemPrompt() != null) {
            conversationTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.SYSTEM, options.systemPrompt())));
        }
    }

    private void invokePrintChat(String str) {
        SwingUtilities.invokeLater(() -> printChat(str));
    }

    private void invokePrintError(Exception ex) {
        SwingUtilities.invokeLater(() -> addError(ex));
    }

    void chat(String userText) {
        inputTextArea.setEnabled(false);
        printUserQuestion(userText + "\n");
        new Thread(() -> {
            try {
                if (state == null) {
                    state = model.createNewState(Llama3.BATCH_SIZE);
                }
                conversationTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, userText)));
                conversationTokens.addAll(chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
                Set<Integer> stopTokens = chatFormat.getStopTokens();
                java.util.List<Integer> responseTokens = Llama.generateTokens(model, state, startPosition,
                        conversationTokens.subList(startPosition, conversationTokens.size()),
                        stopTokens, options.maxTokens(), sampler, options.echo(),
                        token -> {
                            if (options.stream()) {
                                if (!model.tokenizer().isSpecialToken(token)) {
                                    invokePrintChat(model.tokenizer().decode(List.of(token)));
                                }
                            }
                        });
                // Include stop token in the prompt history, but not in the response displayed to the user.
                conversationTokens.addAll(responseTokens);
                startPosition = conversationTokens.size();
                Integer stopToken = null;
                if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
                    stopToken = responseTokens.getLast();
                    responseTokens.removeLast();
                }
                if (!options.stream()) {
                    String responseText = model.tokenizer().decode(responseTokens);
                    invokePrintChat(responseText);
                }
            } catch (Exception ex) {
                invokePrintError(ex);
            } finally {
                printUserQuestion("\n");
                inputTextArea.setEnabled(true);
                inputTextArea.grabFocus();
            }
        }).start();
    }
}
