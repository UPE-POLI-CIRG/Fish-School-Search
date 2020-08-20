package General;
import java.awt.*;
import java.awt.geom.*;
import javax.swing.*;

@SuppressWarnings("serial")
public class Simulation extends JPanel {
	JFrame f;
	Graphics2D g;
	int PAD;
	int SIZE;
	int t = 0;
	
	double[][][] data;
	double range;
	
	public Simulation(double[][][] data, boolean free_particle){
		this.data = data;
		this.range = Run.HIGHEST_SPACE_BOUNDARY;
		if (free_particle) {
			this.PAD  = 250;
			this.SIZE = 1000;
		} else {
			this.PAD  = 10;
			this.SIZE = 800;
		}
		
		JFrame f = new JFrame();
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        f.add(this);
        f.setSize(this.SIZE,this.SIZE);
        f.setLocation(100,100);
        f.setVisible(true);
	}
	
	@SuppressWarnings("unused")
	protected void paintComponent(Graphics g) {
		super.paintComponent(g);
		Graphics2D g2 = (Graphics2D)g;
		g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING,
                        	RenderingHints.VALUE_ANTIALIAS_ON);
		int w = getWidth();
		int h = getHeight();
    
		g2.draw(new Line2D.Double(PAD, PAD, PAD, h-PAD));
		g2.draw(new Line2D.Double(PAD, h-PAD, w-PAD, h-PAD));
		g2.draw(new Line2D.Double(w-PAD, h-PAD, w-PAD, PAD));
		g2.draw(new Line2D.Double(PAD, PAD, w-PAD, PAD));
		g2.setPaint(Color.gray);
		g2.draw(new Line2D.Double(w/2, PAD, w/2, h-PAD));
		g2.draw(new Line2D.Double(PAD, h/2, w-PAD, h/2));
		
		double xScale = (double)(w - 2*PAD)/(2*this.range);
		double yScale = (double)(h - 2*PAD)/(2*this.range);
		
		String alg = Run.algorithmName + " - " + Run.fitnessFunctionName;
		g2.drawString(alg, 20, 30);
		g2.drawString("Iteration " + t, 20, 45);
		
		g2.setPaint(Color.red);
		if(t < data.length){
        	for(int i = 0; i < data[t].length; i++) {
        		
        		double x_m = 0.0;
        		double y_m = 0.0;
        		
        		if(false){
	        		for(int cont = 0; cont < Run.DIMENSIONS; cont++){
	        			if(cont < Run.DIMENSIONS/2)
	        				x_m += data[t][i][cont];
        				else
        					y_m += data[t][i][cont];
	        		}
	        		x_m = x_m/(Run.DIMENSIONS/2);
	        		y_m = y_m/(Run.DIMENSIONS/2);
        		} else {
        			x_m = data[t][i][0];
        			y_m = data[t][i][1];
        		}
        		
        		double x = PAD + (this.range + x_m)*xScale;
        		double y = PAD + (this.range + y_m)*yScale;
        		
        		g2.fill(new Ellipse2D.Double(x-2, y-2, 4, 4));
        	}
        	
        	try {
        	    Thread.sleep(50);
        	    t++;
        	    super.repaint();
        	} catch(InterruptedException ex) {
        	    Thread.currentThread().interrupt();
        	}
		} else {
			for(int i = 0; i < data[t-1].length; i++) {
        		double x = PAD + (this.range + data[t-1][i][0])*xScale;
        		double y = PAD + (this.range + data[t-1][i][1])*yScale;
        		
        		g2.fill(new Ellipse2D.Double(x, y, 4, 4));
        	}
		}
    }
}
