package com.dsp.haptrix;


import android.content.ClipData;
import android.content.ClipDescription;
import android.content.Context;
import android.content.DialogInterface;
import android.content.SharedPreferences;
import android.content.pm.ActivityInfo;
import android.graphics.Color;
import android.graphics.Point;
import android.graphics.drawable.Drawable;
import android.graphics.drawable.GradientDrawable;
import android.graphics.drawable.ShapeDrawable;
import android.graphics.drawable.shapes.RectShape;
import android.graphics.drawable.shapes.Shape;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.support.annotation.ColorInt;
import android.support.constraint.ConstraintLayout;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.text.Html;
import android.text.Spanned;
import android.text.TextUtils;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;

import android.view.Display;
import android.view.DragEvent;
import android.view.View;

import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Scroller;
import android.widget.Toast;
import android.widget.TextView;

import com.android.volley.DefaultRetryPolicy;
import com.android.volley.Request;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.JsonObjectRequest;
import com.android.volley.toolbox.StringRequest;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.UUID;


//import com.google.android.gms.common.AccountPicker;

import android.support.annotation.Nullable;

import static java.lang.Boolean.FALSE;
import static java.lang.Boolean.TRUE;

public class MainActivity extends AppCompatActivity {
//    RequestQueue requestQueue;

//    String url = "http://192.168.0.16";
//    String url = "http://132.206.74.93";
    String url = "http://3.96.14.15"; //aws

    static String APP_PASSCODE;
    static String APP_VERSION;
    Integer NUM_VIBRATIONS;
    Integer MIN_NUM_CLUSTERS;
    Integer TOTAL_MIN_NUM_TIMES_PLAYED = 3; // total

    // Defines details of the stimuli... sorry it is hard coded. I leave as an exercise the parametrization of these variables.
    int vibration_duration = 2; // seconds
    int n_bin = 20;
    int bin_length = vibration_duration * 1000 / n_bin; // mseconds

    String port = "5000";
    String hostname = url + ':' + port;

    boolean amplitude = FALSE;

    int CLUSTER_MIN_NUM_ELEM = 2; // min number of elements in a cluster
    // this value is different from the minimum number of clusters!

    double CLUSTER_MAX_DIST_PX = 300; // max distance

    String post_url = hostname + "/answer";

    // Create empty variables for the vibrations
    int[] hapsig_n = new int[n_bin];
    long[] mVibrateTiming = new long[n_bin];

    private static String uuid_user = null;

    long timer_begin;

//    ShapeDrawable shapedrawable;
    GradientDrawable shapedrawable;
    String completion_code = "";
    String button_pressed_sequence = "";

    String PREF_UNIQUE_ID;

    Integer user_rating_count = 0;

    DefaultRetryPolicy retryPolicy = new DefaultRetryPolicy(
            2000,//750
            1,//0
            1.0f//DefaultRetryPolicy.DEFAULT_BACKOFF_MULT //0.5f
    );
//    private android.widget.RelativeLayout.LayoutParams layoutParams;
    String msg = "debug";
    Integer[] current_memberships;
    String is_group;

    Button[] buttons;
    int[] allVibrationsPlayed;
    boolean[] allTactonsMoved;
    JSONObject current_resp;

    Integer last_button_pressed = -1;


    // onCreate runs when the app is started.
    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_main);
        findViewById(R.id.completion_code).setVisibility(View.INVISIBLE);
        final boolean debug_mode = false;
        final ConstraintLayout constraintlayout = findViewById(R.id.general_constraint_layout);


        // Force portrait application
        this.setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_PORTRAIT);


        TextView counter = findViewById(R.id.countView);
        counter.setText("Count: " + "");

        // Set the instruction box
//        inst.setText("Choose which vibration\n best corresponds to the adjective.");

        final TextView hasvibration = findViewById(R.id.hasvibration);
        final TextView hasamplitudecontrol = findViewById(R.id.hasamplitudecontrol);

        // Disabing the 2 buttons
        hasamplitudecontrol.setAlpha(0);
        hasvibration.setAlpha(0);

        // Instanciates the vibrator
        final Vibrator vibrator = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);

        // Updates on screen content based on whether the phone has vibration...
        if (vibrator.hasVibrator()) {
            hasvibration.setText("HasVibration: TRUE");

        } else {
            hasvibration.setText("HasVibration: FALSE");
            Toast.makeText(MainActivity.this,
                    "This device is too old for running the application, or does not support vibration control.",
                    Toast.LENGTH_LONG).show();
            finish();
        }

        // Updates on screen content AND get_URL depending on whether the phone has amplitude control
        if (Build.VERSION.SDK_INT >= 26) {
            if (vibrator.hasAmplitudeControl()) {
//                hasamplitudecontrol.setText("HasAmplitudeControl: TRUE");
                amplitude = TRUE;
            } else {
//                hasamplitudecontrol.setText("HasAmplitudeControl: FALSE");
            }
        }

        get_version_and_password_from_server();

        // submit button
        final Button submit_button = findViewById(R.id.submit_button);

        submit_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                v.setEnabled(false); // disable button to avoid double-clicking
                String time_taken = String.valueOf(System.currentTimeMillis() - timer_begin);

//                if (check_all_vibrations_played(TOTAL_MIN_NUM_TIMES_PLAYED)
//                        && check_min_num_clusters()
//                        && check_buttons_moved()
////                        && Integer.valueOf(time_taken) > 45000
//                ) {
                    String hapid = null;
                    String question = null;
                    String checksum = null;
                    try {
                        hapid = current_resp.getString("hapid");
                        question = current_resp.getString("question");
                        checksum = current_resp.getString("checksum");
                    }
                    catch (JSONException e){}


                getJsonResponsePost(Arrays.toString(current_memberships),
                        hapid,
                        question,
                        checksum,
                        time_taken);
//                }
//                else{
//                    v.setEnabled(true); // re-enable button
//                    show_submit_message_error();
//                }



            }
        });



        constraintlayout.setOnDragListener(new View.OnDragListener() {
            @Override
            public boolean onDrag(View v, DragEvent event) {
                switch(event.getAction()) {
                    case DragEvent.ACTION_DRAG_STARTED:
                        if (debug_mode) {Log.wtf(msg, "Action is DragEvent.ACTION_DRAG_STARTED");}
                        // Do nothing
                        break;

                    case DragEvent.ACTION_DRAG_ENTERED:
                        if (debug_mode) {Log.wtf(msg, "Action is DragEvent.ACTION_DRAG_ENTERED");}
                        int x_cord = (int) event.getX();
                        int y_cord = (int) event.getY();
                        break;

                    case DragEvent.ACTION_DRAG_EXITED :
                        if (debug_mode) {Log.wtf(msg, "Action is DragEvent.ACTION_DRAG_EXITED");}

//                        x_cord = (int) event.getX();
//                        y_cord = (int) event.getY();

//                        Integer button_number = (int) event.getLocalState();
//                        Button btn_selected = buttons[button_number];
//                        int half_btn_width = btn_selected.getWidth() / 2;
//                        int half_btn_height = btn_selected.getHeight() / 2;
//
//                        btn_selected.setX(x_cord - half_btn_width);
//                        btn_selected.setY(y_cord -  half_btn_height);
                        break;

                    case DragEvent.ACTION_DRAG_LOCATION  :
                        if (debug_mode) {Log.wtf(msg, "Action is DragEvent.ACTION_DRAG_LOCATION");}
                        if (debug_mode) {Log.wtf(msg, " X " + Integer.toString(x_cord) + " Y " + Integer.toString(y_cord));}

                        x_cord = (int) event.getX();
                        y_cord = (int) event.getY();

                        Integer button_number2 = (int) event.getLocalState();
                        Button btn_selected2 = buttons[button_number2];
                        int half_btn_width2 = btn_selected2.getWidth() / 2;
                        int half_btn_height2 = btn_selected2.getHeight() / 2;

                        btn_selected2.setX(x_cord - half_btn_width2);
                        btn_selected2.setY(y_cord -  half_btn_height2);
                        break;

                    case DragEvent.ACTION_DRAG_ENDED:
                        if (debug_mode) {Log.wtf(msg, "Action is DragEvent.ACTION_DRAG_ENDED");}

                        break;

                    case DragEvent.ACTION_DROP:
                        if (debug_mode) {Log.wtf(msg, "ACTION_DROP event");}
                        x_cord = (int) event.getX();
                        y_cord = (int) event.getY();

                        Integer button_number3 = (int) event.getLocalState();
                        Button btn_selected3 = buttons[button_number3];
                        int half_btn_width3 = btn_selected3.getWidth() / 2;
                        int half_btn_height3 = btn_selected3.getHeight() / 2;

                        allTactonsMoved[button_number3] = true;

                        btn_selected3.setX(x_cord - half_btn_width3);
                        btn_selected3.setY(y_cord -  half_btn_height3);
                        ArrayList clusters = get_clusters(buttons);

//                        if (check_all_vibrations_played(1)) { //TODO
                            current_memberships = fill_buttons_colors(buttons, clusters);
//                        }
//                        maybe_enable_submit_button();

                        if (debug_mode) {Log.wtf(msg, Arrays.toString(current_memberships));}

                        break;
                    default: break;
                }
                return true;
            }
        });
    }

    private void delete_buttons(){
        ConstraintLayout constraint_layout = findViewById(R.id.general_constraint_layout);
        try{
        for (int i = 0; i < buttons.length; i++){
            constraint_layout.removeView(buttons[i]);
        }}
        catch(Exception e){}

    }

    public void reset_experiment(){

        delete_buttons();
        button_pressed_sequence = "";
        current_memberships = new Integer[NUM_VIBRATIONS];
        Arrays.fill(current_memberships, -1);
        buttons = new Button[NUM_VIBRATIONS];
        allVibrationsPlayed  = new int[NUM_VIBRATIONS];
        allTactonsMoved = new boolean[NUM_VIBRATIONS];
        timer_begin = System.currentTimeMillis();
        getJsonResponse(NUM_VIBRATIONS);

    }
    private Integer[] fill_buttons_colors(Button[] buttons, ArrayList clusters){
        //        blue_bckgrnd = button_vibrate1.getBackground();
        @ColorInt final int PINK = 0xFFE60ED0;
        @ColorInt final int ORANGE = 0xFFF79605;
        @ColorInt final int NAVY_BLUE = 0xFF0388FC;

        int[] COLORS = {
                Color.YELLOW,
                ORANGE,
                Color.RED,
                Color.CYAN,
                Color.GREEN,
                Color.BLUE,
                Color.MAGENTA,
                NAVY_BLUE,
                PINK,
        };

        Integer[] memberships = new Integer[buttons.length];

        for (int i =0; i<buttons.length; i++){
            shapedrawable = new GradientDrawable();
            shapedrawable.setShape(GradientDrawable.RECTANGLE);

            int cluster_number = find_button_cluster_number(buttons[i], clusters);
            if (cluster_number < 0){ // case of no cluster found by the algo, make gray
                shapedrawable.setColor(Color.LTGRAY);
            }
            else {
                shapedrawable.setColor(COLORS[cluster_number]);
            }
            memberships[i] = cluster_number;
            shapedrawable.setStroke(8, Color.BLACK);
            buttons[i].setBackgroundDrawable(shapedrawable);
        }
        return memberships;
    }

    private ArrayList get_clusters(Button[] buttons){
        float[][] x_y_pairs = new float[NUM_VIBRATIONS][2];

        ArrayList<ArrayList> points = new ArrayList<>();

        for (int i = 0; i<NUM_VIBRATIONS; i++) {
            float X = buttons[i].getX();
            float Y = buttons[i].getY();
            x_y_pairs[i][0] = X;
            x_y_pairs[i][1] = Y;

            ArrayList pair = new ArrayList();
            pair.add(X);
            pair.add(Y);

            points.add(pair);

        }

//        KMeans kmeans = new KMeans();
//
//        List<KMeans.Mean> cluster_means = kmeans.predict(3, x_y_pairs);
//
//        for (int j = 0; j < cluster_means.size(); j++){
//
//            KMeans.Mean mean_obj = cluster_means.get(j);
//            mean_obj.mClosestItems
//        }

//        double[][] dist_mtrx = new double[x_y_pairs.length][x_y_pairs.length];
//        String[] names = new String[x_y_pairs.length];
//
//
//        for (int j = 0; j < x_y_pairs.length; j++){
//            for (int k = 0; k < x_y_pairs.length; k++) {
//                dist_mtrx[j][k] = get_euclidean_distance(x_y_pairs[j][0], x_y_pairs[k][0], x_y_pairs[j][1], x_y_pairs[k][1]);
//            }
//            names[j] = String.valueOf(j);
//        }

//        DBSCAN dbscan = new DBSCAN(300.0, 2);
//
//        dbscan.setPoints(points);
//
//        dbscan.cluster();

        ArrayList result = new ArrayList();
        try{
        DBSCANClusterer dbscan_clusterer = new DBSCANClusterer(points, CLUSTER_MIN_NUM_ELEM, CLUSTER_MAX_DIST_PX, new EuclideanDistanceMetric());

        result = dbscan_clusterer.performClustering();
        }
        catch (DBSCANClusteringException e){
        }

        return result;
    }


    private int find_button_cluster_number(Button button, ArrayList clusters){

        float this_button_x = button.getX();
        float this_button_y = button.getY();

        int cluster_number = -1;

        for (int c = 0; c<clusters.size(); c++){
            int size_ = ((ArrayList) clusters.get(c)).size();
            ArrayList a1 = (ArrayList) clusters.get(c);
            for (int point = 0; point < size_; point++){
                ArrayList a2 = (ArrayList) a1.get(point);

                Float x = (Float) a2.get(0);
                Float y = (Float) a2.get(1);
                if (x == this_button_x && y== this_button_y){
                    cluster_number = c;
                    return cluster_number;
                }
            }
        }
        return -1;
    }




    private void get_version_and_password_from_server() {
        String requesturl = hostname + "/app_info/";
        JsonObjectRequest jsonObjectRequest = new JsonObjectRequest(Request.Method.GET, requesturl, null,
                new Response.Listener<JSONObject>() {

                    @Override
                    public void onResponse(JSONObject response) {
                        try {
                            APP_VERSION = response.getString("app_version");
                            APP_PASSCODE = response.getString("app_passcode");
                            NUM_VIBRATIONS = Integer.valueOf(response.getString("num_vibrations"));
                        }
                        catch (JSONException e){
                            APP_VERSION = "0";
                            APP_PASSCODE = "";
                            NUM_VIBRATIONS = 0;
                        }

                        // Get uuid for Mturk - where getting the email would violate terms of service
                        PREF_UNIQUE_ID = "PREF_UNIQUE_ID_V" + APP_VERSION;

                        SharedPreferences prefs = MainActivity.this.getSharedPreferences("Share", Context.MODE_PRIVATE);

                        if (!prefs.getBoolean("consent_form_accepted" + APP_VERSION, false)) {
                            show_app_password_screen();
                        } else {
                            getUUID_NO_EMAIL(getApplicationContext());
                            // START EXPERIMENT WHERE WE LEFT OFF.
//                            reset_experiment();
                        }




                    }
                }, new Response.ErrorListener() {

            @Override
            public void onErrorResponse(VolleyError error) {
            }
        });
        jsonObjectRequest.setShouldCache(false);
        jsonObjectRequest.setRetryPolicy(retryPolicy);
        RequestQueueSingleton.getInstance(this).addToRequestQueue(jsonObjectRequest);
    }

    private void show_app_password_screen() {
        final EditText txtUrl = new EditText(this);
        txtUrl.setSingleLine(true);

        txtUrl.setHint("Enter password for the app.");

        new AlertDialog.Builder(this)
                .setTitle("RateVibrations V3")
                .setView(txtUrl)
                .setCancelable(false)
                .setPositiveButton("go", new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int whichButton) {
                        String passcode_given = txtUrl.getText().toString();
                        if (passcode_given.equals(APP_PASSCODE)) {
                            show_consent_form();
                        } else {
                            show_app_password_screen();
                        }
                    }
                })
                .show();
    }

    public void show_explanation_screen() {

        final AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle(Html.fromHtml("<font color=red>Important note</font>"));

        String MSG = "Through this experiment we are trying to learn what makes people consider two vibrations as being similar.<br /><br />" +
                "We are NOT asking you to tell us whether the vibrations are the SAME, " +
                "instead we are looking for your <b>gut feeling as to whether they feel similar to you</b>.<br />" +
                "A synonym to 'similar' would be 'close', 'comparable', 'near', 'alike', or 'resembling without being identical'.<br /><br />" +
                "We will start by showing you a few examples of what we are looking for in a 'calibration' phase, " +
                "and then proceed with the experiment.<br />" +
                "We ask that you pay attention during this calibration phase. Thank you!";

        String MSG_NO_CALIB = "Through this experiment we are trying to learn what makes people consider two vibrations as being similar.<br /><br />" +
                "We are NOT asking you to tell us whether the vibrations are the SAME, " +
                "instead we are looking for your <b>gut feeling as to whether they feel similar to you</b>.<br />" +
                "A synonym to 'similar' would be 'close', 'comparable', 'near', 'alike', or 'resembling without necessarily being identical'.<br /><br />" +
                "Enjoy!";

        builder.setMessage(Html.fromHtml(MSG_NO_CALIB));
        builder.setPositiveButton(android.R.string.yes, new DialogInterface.OnClickListener() {
            public void onClick(DialogInterface dialog, int which) {
                dialog.dismiss();
            }
        });
        builder.setIcon(android.R.drawable.ic_dialog_info);
        builder.setCancelable(false);
        final AlertDialog dlg = builder.create();
        dlg.setCancelable(false);
        dlg.setCanceledOnTouchOutside(false);

        WindowManager.LayoutParams lp = new WindowManager.LayoutParams();
        lp.copyFrom(dlg.getWindow().getAttributes());
        lp.width = WindowManager.LayoutParams.MATCH_PARENT;
        lp.height = WindowManager.LayoutParams.MATCH_PARENT;
        dlg.show();
        dlg.getWindow().setAttributes(lp);

        TextView textView = (TextView) dlg.findViewById(android.R.id.message);
        textView.setScroller(new Scroller(this));
        textView.setVerticalScrollBarEnabled(true);
        textView.setMovementMethod(new ScrollingMovementMethod());
        textView.setVerticalScrollBarEnabled(true);
    }

    public void show_consent_form() {

        final AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Participant consent form");

        String MSG = "McGill University REB File # 432-0416\n" +
                "<b><u>Evaluating the effects of vibration characteristics on perception of affective tactile interactions</u></b> <br />" +
                "<b>The purpose of the study</b> is to establish the relationship between vibration signal characteristics and " +
                "sensation. The results will be used to increase the efficiency of affective and wearable communication " +
                "interfaces (e.g., smartwatches, smart shoes, etc.).<br />" +

                "<b>The study procedures</b> involve the use of your smartphone in comparing different vibrations coming " +
                "from a vibration-generating device. You will be presented with vibrations and asked to report on how " +
                "you perceive various attributes of these vibrations using your personal smartphone.<br />" +
                "<b>Participation in this study is voluntary.</b> You may refuse to participate in any part of the study, " +
                "and may withdraw from the study at any time, for any reason. You are " +
                "free to stop participation in this study at any time, and to request that your data be deleted after you " +
                "have completed the experiment. We cannot delete your data before completion of the experiment, as all " +
                "data collected is anonymized and cannot be traced back to you until you submit.<br />" +
                "For participants recruited via Amazon Mechanical Turk: <br />" +
                "You may email the Amazon Requester to ask that your data be wiped from our database, but this cannot " +
                "be processed if the experiment is not done or if the payment has already been sent to the worker.<br />" +
                "The <b>potential risks</b> of this study are minimal. The experimental activities consist of perceiving " +
                "vibrations in the palm of your hand that holds the smartphone, which are typical of the strength of " +
                "smartphone notifications. You are advised to take breaks from the study if fatigue is a concern. " +
                "Participating in the study might not <b>benefit you</b>, but we hope to learn from this research how different" +
                "vibrations affect our reactions to them.<br />" +
                "Since this is an in-the-wild study, you will run the study app at one or more time(s) of your choosing " +
                "throughout the day. It is expected that the total time you spend entering inputs via the study’s " +
                "smartphone app will not exceed 60 minutes, whether consecutively or spread out over the day. How " +
                "you break down the time between the ratings is up to you.<br />" +
                "<b>You will be compensated</b> pro rata $1 USD per 60 ratings for AMT workers. If your ratings satisfy a validation measure, you " +
                "will be awarded a bonus of $1 USD and may be invited to participate in subsequent rounds of the study. We " +
                "will proceed with the payment within 48 hours of your completion of the HIT. Non-AMT workers will be compensated in an 80%-20% scheme payment.<br />" +
                "<b>Confidentiality:</b> We will not collect any personal identifying information from you during the " +
                "experiment. We will not reveal your identity to anyone outside of the experimenters and their " +
                "supervisor. The data collected during this experiment will be used only in resulting publications. " +
                "You will be assigned a random ID number, and all your collected data will be associated only with this " +
                "ID. Only the research investigators and their supervisor will have access to the raw data, which will be " +
                "kept securely on a password protected computer. You will be referred to only by subject ID in any " +
                "resulting publications.<br />" +
                "For participants recruited via Amazon Mechanical Turk:<br />" +
                "We collect your worker ID only for the purpose of payment.<br />" +
                "All data will be saved for 7 years following publication to comply with McGill requirements." +
                "For participants recruited via Amazon Mechanical Turk:<br />" +
                "By accepting the HIT through the Mechanical Turk website, you consent to the above conditions." +
                "Should you have any questions about this study, you may contact the research supervisor, Prof. Jeremy " +
                "Cooperstock at jer@cim.mcgill.ca or by telephone at 514-398-5992.<br />" +
                "If you have any ethical concerns or complaints about your participation in this study, and want " +
                "to speak with someone not on the research team, please contact the McGill Ethics Manager at " +
                "514-398-6831 or lynda.mcneil@mcgill.ca.";



        builder.setMessage(Html.fromHtml(MSG));
        builder.setPositiveButton(android.R.string.yes, new DialogInterface.OnClickListener() {
            public void onClick(DialogInterface dialog, int which) {
                SharedPreferences prefs = builder.getContext().getSharedPreferences("Share", Context.MODE_PRIVATE);
                SharedPreferences.Editor editor = prefs.edit();
                editor.putBoolean("consent_form_accepted" + APP_VERSION, true); // the variable name has the app version so that when we switch to new app we have a new consent form
                editor.apply();
                getUUID_NO_EMAIL(getApplicationContext());
                dialog.dismiss();
                show_explanation_screen();
            }
        });
        builder.setIcon(android.R.drawable.ic_dialog_info);
        builder.setCancelable(false);
        final AlertDialog dlg = builder.create();
        dlg.setCancelable(false);
        dlg.setCanceledOnTouchOutside(false);

        WindowManager.LayoutParams lp = new WindowManager.LayoutParams();
        lp.copyFrom(dlg.getWindow().getAttributes());
        lp.width = WindowManager.LayoutParams.MATCH_PARENT;
        lp.height = WindowManager.LayoutParams.MATCH_PARENT;
        dlg.show();
        dlg.getWindow().setAttributes(lp);


        TextView textView = (TextView) dlg.findViewById(android.R.id.message);
//        textView.setMaxLines(15);
        textView.setScroller(new Scroller(this));
        textView.setVerticalScrollBarEnabled(true);
        textView.setMovementMethod(new ScrollingMovementMethod());
        textView.setVerticalScrollBarEnabled(true);




    }

    private long[] toArray(List<Long> values) {
        long[] result = new long[values.size()];
        int i = 0;
        for (Long l : values)
            result[i++] = l;
        return result;
    }

    private int[] itoArray(List<Integer> values) {
        int[] result = new int[values.size()];
        int i = 0;
        for (int l : values)
            result[i++] = l;
        return result;
    }

    // function that presents vibrations
    public void present_vibrations(long[] mVibratePattern, int[] mAmplitudes) {
        final Vibrator vibrator = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);
        // Binarize the pattern with threshold at 50% intensity
        List<Long> list = new ArrayList<Long>();
        List<Integer> amps = new ArrayList<Integer>();
        long duration = 0;
        int previous = -1;
        for (int j : mAmplitudes) {
            if (previous == -1) {
                if (j == 255) {
                    list.add(duration);
                    amps.add(0);
                }
                duration += bin_length;

            } else {
                if (previous != j) {
                    list.add(duration);
                    amps.add(previous);
                    duration = bin_length;
                } else duration += bin_length;
            }
            previous = j;
        }
        list.add(duration);
        amps.add(previous);
        final long[] dumb_amplitude = toArray(list);
        final int[] ampl = itoArray(amps);


        // If device has amplitude control, present the stimulus directly as received
        if (Build.VERSION.SDK_INT >= 26) {
            VibrationEffect effect = VibrationEffect.createWaveform(dumb_amplitude, ampl, -1);
            vibrator.vibrate(effect);
        } else vibrator.vibrate(dumb_amplitude, -1);

    }

    private void toggle_enabledness_buttons(boolean toggle) {
        try{
        for (int i = 0; i < buttons.length; i++) {
            buttons[i].setEnabled(toggle);
        }}
        catch (NullPointerException e) {}

//        findViewById(R.id.submit_button).setEnabled(toggle);
    }

    private void toggle_visibility_buttons(boolean toggle) {
        int visible = View.VISIBLE;
        if (!toggle) {visible = View.INVISIBLE;}

        try{
            for (int i = 0; i < buttons.length; i++) {
                buttons[i].setVisibility(visible);
            }}
        catch (NullPointerException e) {}

        findViewById(R.id.submit_button).setVisibility(visible);
    }

    private void decypher_get_response(JSONObject resp){
        try{
            current_resp = resp;
            String checksum = resp.getString("checksum");
            String hapid = resp.getString("hapid");
            String q_asked = resp.getString("question");
            JSONArray signals = resp.getJSONArray("hapsig"); //use signal.get(index)
            MIN_NUM_CLUSTERS = Integer.valueOf(resp.getString("min_num_clusters"));
            is_group = resp.getString("is_group");

            // Updates Question on screen
            TextView inst = findViewById(R.id.instruction);
            Spanned mm = Html.fromHtml("Group the following vibrations" +
                    " into <br /><u>at least</u> <big><b> "+String.valueOf(MIN_NUM_CLUSTERS)+" </b></big> " +
                    "groups <br />according to their similarity.");
            inst.setText(mm);

            // Create array of N buttons, display them on screen
            final int minX = 150;
            final int minY = 250;

            // Get the screen specifications
            Display display = getWindowManager().getDefaultDisplay();
            Point size = new Point();
            display.getSize(size);
            final int maxX = size.x;
            final int maxY = size.y;

            // set the max dist based on screen specs
            CLUSTER_MAX_DIST_PX = size.y / 7f;


            final ConstraintLayout constraintlayout = findViewById(R.id.general_constraint_layout);

            final Button button_template = findViewById(R.id.button_template);
            button_template.setVisibility(View.INVISIBLE);
            ViewGroup.LayoutParams base_layout_params = button_template.getLayoutParams();

            for (int i = 0; i < NUM_VIBRATIONS; i++){
                final Button but = new Button(this);
                but.setLayoutParams(base_layout_params);

                float randomX = new Random().nextInt((maxX - minX) + 1) + minX;
                float randomY = new Random().nextInt((maxY - minY) + 1) + minY;

                if (i < (NUM_VIBRATIONS+1)/2){
                    randomX = minX;
                }
                else{
                    randomX = minX + ((maxX - minX) / 2);
                }
                randomY = (i%((NUM_VIBRATIONS+1)/2)) * (maxY/8) + minY;

//            but.setX((i % 2) * 500 + 200);
//            but.setY((i % 5) * 300 + 300);
                but.setX(randomX);
                but.setY(randomY);

                String button_text = String.valueOf(i + 1); //our db has 0->N-1 , we need to map from 1->N
                but.setText(button_text);
                but.setVisibility(View.VISIBLE);
                final int local_state = i;
                but.setOnLongClickListener(new View.OnLongClickListener() {
                    @Override
                    public boolean onLongClick(View v) {
                        ClipData.Item item = new ClipData.Item((CharSequence)v.getTag());
                        String[] mimeTypes = {ClipDescription.MIMETYPE_TEXT_PLAIN};

                        ClipData dragData = new ClipData("some_tag", mimeTypes, item);
                        View.DragShadowBuilder myShadow = new View.DragShadowBuilder(button_template);
                        myShadow.getView().setAlpha(0);
                        v.startDragAndDrop(dragData,myShadow,local_state,View.DRAG_FLAG_OPAQUE); // drag flag opaque will make the shadow opaque
                        return true;
                    }
                });
                final JSONArray sig = (JSONArray) signals.get(i);
                final String but_pressed = String.valueOf(i);

                but.setOnClickListener(new View.OnClickListener() {
                    // This happens when the vibration button is clicked
                    public void onClick(View v) {
                        int k = Integer.valueOf(but_pressed);

                        // pressing the same button in a row does not count as multiple presses (counts only as one)
                        if (k != last_button_pressed){
                            allVibrationsPlayed[k] = allVibrationsPlayed[k] + 1;
                        }

                        presentVib(sig);

                        button_pressed_sequence = button_pressed_sequence + "|" + but_pressed;
                        last_button_pressed = k;

                        }
                    }
                );

                buttons[i] = but;
                allVibrationsPlayed[i] = 0;
                allTactonsMoved[i] = false;

                // Color & shape
                shapedrawable = new GradientDrawable();
                shapedrawable.setShape(GradientDrawable.RECTANGLE);
                shapedrawable.setColor(Color.LTGRAY);
                shapedrawable.setStroke(8, Color.BLACK);
                but.setBackgroundDrawable(shapedrawable);

                constraintlayout.addView(but);

            }

        }catch (JSONException e){
            Toast.makeText(getApplicationContext(), "error parsing response", Toast.LENGTH_LONG);
            this.toggle_enabledness_buttons(true);
        }
    }

    private boolean check_all_vibrations_played(Integer min_num_times_played){

        for (int m = 0; m < NUM_VIBRATIONS; m++){
            if (allVibrationsPlayed[m] < min_num_times_played){
                return false;
            }
        }
        return true;
    }

    private boolean check_buttons_moved(){
        int count = 0;
        for (int m = 0; m < NUM_VIBRATIONS; m++){
            count += (allTactonsMoved[m] ? 1 : 0);
            }

        return count >= Math.round(0.75 * NUM_VIBRATIONS);
    }

    private boolean check_min_num_clusters(){

        return Collections.max(Arrays.asList(current_memberships)) >= MIN_NUM_CLUSTERS-1; //-1 because starts at 0
//        for (int m = 0; m < current_memberships.length; m++){
//            if (current_memberships[m]){
//                return false;
//            }
//        }
//        return true;
    }

    private void presentVib(JSONArray signal) {

        try {
            toggle_enabledness_buttons(false);

            for (int k = 0; k < signal.length(); k++) {
                hapsig_n[k] = (int) signal.get(k);
            }

            Arrays.fill(mVibrateTiming, bin_length);

            findViewById(R.id.submit_button).setEnabled(false);

            // Attempts to present vibrations
            present_vibrations(mVibrateTiming, hapsig_n);

            new Handler().postDelayed(new Runnable() { // This is used to ensure an acceptable delay between when you click on the button and the vibrations presented, before activating the answer buttons
                @Override
                public void run() {
                    toggle_enabledness_buttons(true);
                    findViewById(R.id.submit_button).setEnabled(true);
//                    maybe_enable_submit_button();
                }
            }, 2000);
        }
        catch (JSONException e){

        }
    }

    private void maybe_enable_submit_button(){ // deprecated because the submit button now displays a message
        // check if all vibrations have been played and if the minimum number of clusters is respected
        // if yes, enable the submit button
        if (check_all_vibrations_played(3) && check_min_num_clusters()) {
            findViewById(R.id.submit_button).setEnabled(true);
        }
        else {
            findViewById(R.id.submit_button).setEnabled(false);
        }

    }

    private void show_submit_message_error(){
        final AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle(Html.fromHtml("<font color=red>Error</font>"));

        // get the problematic vibrations that cause the error
        List<String> problems = new ArrayList<>();

        for (int i = 0; i < NUM_VIBRATIONS; i++){
            if (allVibrationsPlayed[i] != TOTAL_MIN_NUM_TIMES_PLAYED || !allTactonsMoved[i]){
                problems.add(String.valueOf(i + 1));// vibration index starts at 0 so we must add 1 to get the vibration number
            }
        }

        String problematic_vibrations = TextUtils.join(", ", problems);


        String MSG;
//        if (MIN_NUM_CLUSTERS > 1) {
        MSG = "To submit you must comply with the following conditions:<br />" +
                "<big>1</big>. All vibrations must have been played at least 3 times (in-a-row clicks count for a single one).<br />" +
                "<big>2</big>. The grouping rule at the top of the screen must be respected: " +
                "you must have at least " + String.valueOf(MIN_NUM_CLUSTERS) + " colored groups.<br />" +
                "<big>3</big>. Most buttons (approx. 75%) must have been moved around.<br />" +
                "Problematic vibration numbers: <br />" + problematic_vibrations;

//        }
//        else {
//            MSG = "To submit you must comply with the following two conditions:<br />" +
//                    "<big>1</big>. All vibrations must have been played at least once.<br />" +
//                    "<big>2</big>. The grouping rule at the top of the screen must be respected " +
//                    "such that all buttons are assigned to a group (in this round, single button groups" +
//                    "are allowed).";
//        }

        builder.setMessage(Html.fromHtml(MSG));
        builder.setPositiveButton(android.R.string.ok, new DialogInterface.OnClickListener() {
            public void onClick(DialogInterface dialog, int which) {
                dialog.dismiss();
            }
        });
        builder.setIcon(android.R.drawable.ic_dialog_info);
        builder.setCancelable(false);
        final AlertDialog dlg = builder.create();
        dlg.setCancelable(false);
        dlg.setCanceledOnTouchOutside(false);
        dlg.show();

//        WindowManager.LayoutParams lp = new WindowManager.LayoutParams();
//        lp.copyFrom(dlg.getWindow().getAttributes());
//        lp.width = WindowManager.LayoutParams.MATCH_PARENT;
//        lp.height = WindowManager.LayoutParams.MATCH_PARENT;
//        dlg.getWindow().setAttributes(lp);

//        TextView textView = (TextView) dlg.findViewById(android.R.id.message);
//        textView.setScroller(new Scroller(this));
//        textView.setVerticalScrollBarEnabled(true);
//        textView.setMovementMethod(new ScrollingMovementMethod());
//        textView.setVerticalScrollBarEnabled(true);
    }

    protected void UUAddUserToDB() {
        String requesturl = hostname + "/id/" + Build.MANUFACTURER + "/" + Build.MODEL + "/" + Boolean.toString(amplitude) + "/" + uuid_user;
        JsonObjectRequest jsonObjectRequest = new JsonObjectRequest(Request.Method.GET, requesturl, null,
                new Response.Listener<JSONObject>() {

                    @Override
                    public void onResponse(JSONObject response) {
                        try {

                            String uuid_user_verification = response.getString("uuid");
                            if (!uuid_user_verification.equals(uuid_user))
                                throw new AssertionError("uuid users not matching in database");
//                            Toast.makeText(MainActivity.this, "UUID_MATCHING_APPROVED", Toast.LENGTH_SHORT).show();

                        } catch (JSONException e) {
//                            Toast.makeText(MainActivity.this, "Wrong Json received", Toast.LENGTH_SHORT).show();
                        }
                    }
                }, new Response.ErrorListener() {

            @Override
            public void onErrorResponse(VolleyError error) {
            }
        });
        jsonObjectRequest.setShouldCache(false);
        jsonObjectRequest.setRetryPolicy(retryPolicy);
        RequestQueueSingleton.getInstance(this).addToRequestQueue(jsonObjectRequest);
    }


    // Using the UUID, get the current count of ratings
    protected void getUUIDRatingsCount() {
        String requesturl = hostname + "/counter/" + uuid_user;
        JsonObjectRequest jsonObjectRequest = new JsonObjectRequest(Request.Method.GET, requesturl, null,
                new Response.Listener<JSONObject>() {

                    @Override
                    public void onResponse(JSONObject response) {
                        try {
                            user_rating_count = Integer.valueOf(response.getString("count"));
                            Integer total_num_calibrations = Integer.valueOf(response.getString("total_num_calibrations"));

                            Integer max_count = Integer.valueOf(response.getString("max_count"));
                            TextView counter = findViewById(R.id.countView);
                            completion_code = response.getString("last_code_given");

                            if (user_rating_count < 0){
                                counter.setText("Count: " + max_count + " out of " + max_count);
                                finalize_HIT(max_count);
                                toggle_enabledness_buttons(false);
                                toggle_visibility_buttons(false);
                            }
                            else if (user_rating_count < (max_count)) {
                                counter.setText("Count: " + (user_rating_count - total_num_calibrations) + " out of " + max_count);
                                reset_experiment();
                            }


                        } catch (JSONException e) {
//                            Toast.makeText(MainActivity.this, "Wrong Json received", Toast.LENGTH_SHORT).show();
                        }
                    }
                }, new Response.ErrorListener() {

            @Override
            public void onErrorResponse(VolleyError error) {
//                Toast.makeText(MainActivity.this, "" + error.toString(), Toast.LENGTH_SHORT).show();
            }
        });

        jsonObjectRequest.setShouldCache(false);
        jsonObjectRequest.setRetryPolicy(retryPolicy);
        RequestQueueSingleton.getInstance(this).addToRequestQueue(jsonObjectRequest);

    }

    public synchronized boolean getUUID_NO_EMAIL(Context context) { // this is called every time the app is active
        SharedPreferences sharedPrefs = context.getSharedPreferences(
                PREF_UNIQUE_ID, Context.MODE_PRIVATE);
        String app_version_registered = sharedPrefs.getString("APP_V", "");
        uuid_user = sharedPrefs.getString("UUID_V", null);
        Log.wtf("UUID1", uuid_user);
        if (uuid_user == null){
            get_version_and_password_from_server();
            Log.wtf("UUID", uuid_user);
        }

        if (app_version_registered.equals(APP_VERSION) && uuid_user != null){
            this.getUUIDRatingsCount();
        }
        else { // generate new uuid
            String uuid_user_uuid = UUID.randomUUID().toString();
            uuid_user = uuid_user_uuid+ "_" + APP_VERSION;
            SharedPreferences.Editor editor = sharedPrefs.edit();
            editor.putString("UUID_V", uuid_user);
            editor.putString("APP_V", APP_VERSION);
            editor.apply();
            this.AddUserToDB();
        }
        return true;
    }
//
    // Request a new stimulus to server
    public void getJsonResponse(final Integer number_stimuli) {
        String tmp = "0";
        //if(amplitude) tmp = "1"; //TODO activate this to activate amplitude modulation.
        String requesturl = hostname + "/pattern/" + uuid_user + "/" + tmp + "/" + Integer.toString(n_bin) + "/" + Integer.toString(number_stimuli) + "/" + Integer.toString(user_rating_count);
        JsonObjectRequest jsonObjectRequest = new JsonObjectRequest(Request.Method.GET, requesturl, null,
                new Response.Listener<JSONObject>() {

                    @Override
                    public void onResponse(JSONObject response) {
                        decypher_get_response(response);
                        findViewById(R.id.submit_button).setEnabled(true);
                    }
                }, new Response.ErrorListener() {

            @Override
            public void onErrorResponse(VolleyError error) {
                // if error, make sure sending to server is deactivated
                try{
                toggle_enabledness_buttons(true);

                }
                catch (Exception e){}

//                findViewById(R.id.but_1).setEnabled(false);
//                get_version_and_password_from_server();
                Toast.makeText(MainActivity.this,
                        "Connection error. Make sure you have internet access and reload the app.",
                        Toast.LENGTH_LONG).show();
            }

        });

        jsonObjectRequest.setShouldCache(false);
        jsonObjectRequest.setRetryPolicy(retryPolicy);
        RequestQueueSingleton.getInstance(this).addToRequestQueue(jsonObjectRequest);

    }

    public void finalize_HIT(Integer max_count) {
        // Disable all buttons + write code on screen
        toggle_enabledness_buttons(false);
        toggle_visibility_buttons(false);

        final TextView completion_code_textview = (TextView) findViewById(R.id.completion_code); // question asked on top
        final TextView instruct = (TextView) findViewById(R.id.instruction);

        if (completion_code == null) {
            completion_code = "";
        }

        TextView counter = findViewById(R.id.countView);
        counter.setText("Count: " + max_count + " out of " + max_count);

        completion_code_textview.setVisibility(View.VISIBLE);
        completion_code_textview.setTextSize(20);
        completion_code_textview.setText(Html.fromHtml("CODE: <br><u>" + completion_code + "</u>"));
        instruct.setText("Here is your completion code.\nThank you for participating!\nWe hope that you enjoyed the experiment.");
    }


    // Posts a response to server
    public void getJsonResponsePost(final String answer, final String hapid, final String question, final String checksum, final String time_taken) {

        StringRequest stringRequest = new StringRequest(Request.Method.POST, post_url,
                new Response.Listener<String>() {
                    @Override
                    public void onResponse(String response) {
                        try {
                            JSONObject resp_json = new JSONObject(response);

                            user_rating_count = Integer.valueOf(resp_json.getString("user_rating_count"));
                            final Integer max_count = Integer.valueOf(resp_json.getString("max_count"));
                            completion_code = resp_json.getString("code"); //update the completion code
                            final Integer total_num_calibrations = Integer.valueOf(resp_json.getString("total_num_calibrations"));

                            // check if the user_rating_count is bigger than a certain number
                            if (user_rating_count < 0) {
                                // Disable all buttons + write code on screen
                                finalize_HIT(max_count);

                            }
                            else {
                                // setting UI state and preparing for next rating
                                TextView counter = findViewById(R.id.countView);
                                counter.setText("Count: " + (user_rating_count - total_num_calibrations) + " out of " + (max_count));
                                reset_experiment();

                            }
                        } catch (JSONException err) {
                            Log.wtf("ERROR", "couldnt convert to json object");
                        }
                    }
                },
                new Response.ErrorListener() {
                    @Override
                    public void onErrorResponse(VolleyError error) {
                        // if error, make sure buttons stay enabled
                        toggle_enabledness_buttons(true);
                        findViewById(R.id.submit_button).setEnabled(true);

                        try {
                            Log.wtf("JSON_POST", "onErrorResponse");
                            String responseBody = new String(error.networkResponse.data, "utf-8");
                            JSONObject data = new JSONObject(responseBody);
                            JSONArray errors = data.getJSONArray("errors");
                            JSONObject jsonMessage = errors.getJSONObject(0);
                            String message = jsonMessage.getString("message");
//                            Toast.makeText(getApplicationContext(), message, Toast.LENGTH_LONG).show();

                        } catch (JSONException e) {
                        } catch (UnsupportedEncodingException error2) {
                        }
                        catch (java.lang.NullPointerException error3) { // This error is raised when there is no connection or a connection error
                            Toast.makeText(MainActivity.this,
                                    "Connection error. Make sure you have internet access and retry.",
                                    Toast.LENGTH_LONG).show();
                        }
                    }
                }) {

            // This is where the information that is being sent it packaged
            @Override
            public Map<String, String> getParams() {
                Map<String, String> params = new HashMap<String, String>();
                params.put("UUID", uuid_user);
                params.put("answer", answer);
                params.put("hapid", hapid);
                params.put("question", question);
                params.put("checksum", checksum);
                params.put("time_taken", time_taken);
                params.put("button_pressed_sequence", button_pressed_sequence);
                params.put("is_group", is_group);

                return params;
            }
        };

        stringRequest.setRetryPolicy(retryPolicy);
        stringRequest.setShouldCache(false);
        RequestQueueSingleton.getInstance(this).addToRequestQueue(stringRequest);

    }
}
