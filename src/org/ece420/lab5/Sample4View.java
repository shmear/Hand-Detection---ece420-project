package org.ece420.lab5;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.core.Point;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import java.util.ArrayList;
import java.util.List;

class Sample4View extends SampleViewBase {
    private static final String TAG = "OCVSample::View";

    public static final int     VIEW_MODE_GRAY     = 0;
    public static final int     VIEW_MODE_RGBA     = 1;
    public static final int     VIEW_MODE_RUN      = 2;
    public static final int 	INITIALIZE         = 1;
    public static final int 	VIEW_MODE_BACKGROUND = 3;

    private Mat                 mYuv;
    private Mat 				back; 
    private Mat                 mRgba;
    private Mat					mHandSeg_RGB;
    private Mat                 mGraySubmat;
    private Mat					dst_gray;
    private Bitmap              mBitmap;
    private int                 mViewMode;
    private int 				background_init;
    private Mat				NumbFingerTips;
    private String 			fingertext;
    private int				NumbFingers;
    private int 			viewmode;

    public Sample4View(Context context) {
        super(context);
    }

    @Override
    protected void onPreviewStarted(int previewWidth, int previewHeight) {
        Log.i(TAG, "called onPreviewStarted("+previewWidth+", "+previewHeight+")");

        // initialize Mats before usage
        mYuv = new Mat(getFrameHeight() + getFrameHeight() / 2, getFrameWidth(), CvType.CV_8UC1);
        mGraySubmat = mYuv.submat(0, getFrameHeight(), 0, getFrameWidth());

        // allocate space now because are using our own color conversion function
        mRgba = new Mat(getFrameHeight(), getFrameWidth(), CvType.CV_8UC4);
        back = new Mat(getFrameHeight(), getFrameWidth(), CvType.CV_8UC4);
        mHandSeg_RGB= new Mat(getFrameHeight(), getFrameWidth(), CvType.CV_8UC4);
        NumbFingerTips = new Mat(5,5,CvType.CV_8UC1);
        fingertext = new String();
        mBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
    }

    @Override
    protected void onPreviewStopped() {
        Log.i(TAG, "called onPreviewStopped");

        if (mBitmap != null) {
            mBitmap.recycle();
            mBitmap = null;
        }

        synchronized (this) {
            // Explicitly deallocate Mats
            if (mYuv != null)
                mYuv.release();
            if (mRgba != null)
                mRgba.release();
            if (mGraySubmat != null)
                mGraySubmat.release();
            if (mHandSeg_RGB != null)
                mHandSeg_RGB.release();

            mYuv = null;
            mRgba = null;
            mGraySubmat = null;
            mHandSeg_RGB = null;
        }

    }


    @Override
    protected Bitmap processFrame(byte[] data) {
    	// data from camera is in YUV420sp format
        mYuv.put(0, 0, data);

        final int viewMode = mViewMode;

        switch (viewMode) {
        case VIEW_MODE_GRAY:
        	// opencv's color conversion function
            Imgproc.cvtColor(mGraySubmat, mRgba, Imgproc.COLOR_GRAY2RGBA, 4);
            break;
        case VIEW_MODE_RGBA:
            Imgproc.cvtColor(mYuv, mRgba, Imgproc.COLOR_YUV420sp2RGB, 4);
            viewmode = 0;
            NumbFingers = HandSegment(mRgba.getNativeObjAddr(),mGraySubmat.getNativeObjAddr(),viewmode);
            Imgproc.cvtColor(mGraySubmat, mRgba, Imgproc.COLOR_GRAY2RGBA, 4);
        	// implement the color conversion manually
        	//YUV2RGB(mYuv.getNativeObjAddr(),mRgba.getNativeObjAddr());
            break;
        case VIEW_MODE_RUN:
        	//Imgproc.equalizeHist(mGraySubmat, mGraySubmat);
        	Imgproc.cvtColor(mYuv, mRgba, Imgproc.COLOR_YUV420sp2RGB, 4);
        	viewmode = 1;
        	
        	//Imgproc.cvtColor(mRgba, mRgba, Imgproc.COLOR_RGB2YCrCb, 4);
        	
        	if (background_init == 1){
        		back = mHandSeg_RGB;
        	}
        	
        	NumbFingers = HandSegment(mRgba.getNativeObjAddr(),mGraySubmat.getNativeObjAddr(),viewmode);
        	switch(NumbFingers){
        	
        	case 0: fingertext = "Nothing";
        			break;
        	case 1: fingertext = "1 Finger";
        			break;
        	case 2: fingertext = "2 Fingers";
        			break;
        	case 3: fingertext = "3 Fingers";
        			break;
        	case 4: fingertext = "4 Fingers";
        			break;
        	case 5: fingertext = "5 Fingers";
        			break;
        	case 6: fingertext = "Fist";
        			break;
        	default: fingertext = "Invalid";
        			break;
        	
        	}
        	/*dst_gray = mGraySubmat;
        	Imgproc.dilate(mGraySubmat, dst_gray, new Mat(5,5,0));
        	Imgproc.medianBlur(dst_gray, mGraySubmat, 1);
        	Imgproc.Canny(mGraySubmat, dst_gray, 50, 255);
        	
        	List<MatOfPoint> contour = new ArrayList<MatOfPoint>();
        	Mat mHierarchy = new Mat();
        	Imgproc.findContours(dst_gray, contour, mHierarchy, 1, 2);
        	mGraySubmat = dst_gray;*/
        	
        	//Imgproc.cvtColor(mGraySubmat, mRgba, Imgproc.COLOR_GRAY2RGBA, 4);
        	background_init = 0;
        	Core.putText(mRgba, fingertext, new Point(getFrameWidth()/2-100, getFrameHeight()-20),Core.FONT_HERSHEY_SIMPLEX, 1.5, new Scalar(255, 0,0,255),3);
        	
        	// apply equalization to Y channel and convert to RGB
            //HistEQ(mYuv.getNativeObjAddr(), mRgba.getNativeObjAddr());
            break;
        case VIEW_MODE_BACKGROUND:
        	viewmode = 2;
        	Imgproc.cvtColor(mYuv, mRgba, Imgproc.COLOR_YUV420sp2RGB, 4);
        	NumbFingers = HandSegment(mRgba.getNativeObjAddr(),mGraySubmat.getNativeObjAddr(),viewmode);
        	
        	
        	break;
        }

        Bitmap bmp = mBitmap;

        try {
            Utils.matToBitmap(mRgba, bmp);
        } catch(Exception e) {
            Log.e("org.opencv.samples.puzzle15", "Utils.matToBitmap() throws an exception: " + e.getMessage());
            bmp.recycle();
            bmp = null;
        }

        return bmp;
    }

	//public native void YUV2RGB(long matAddrYUV, long matAddrRgba);
    //public native void HistEQ(long matAddrYUV, long matAddrRgba);
    public native int HandSegment(long matAddrRgba, long matAddrHandSegment, int view);

    public void setViewMode(int viewMode) {
        Log.i(TAG, "called setViewMode("+viewMode+")");
        mViewMode = viewMode;
    }
    
    public void setinit(int Init) {
    	 Log.i(TAG, "called setinit("+Init+")");
    	 background_init = Init;
    }
}
