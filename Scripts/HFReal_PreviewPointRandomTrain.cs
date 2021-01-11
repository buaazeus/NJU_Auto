using System.Collections.Generic;
using UnityEngine;
using MLAgents;
using EVP;
using System;
using System.IO;
using System.Text.RegularExpressions;
using UnityEngine.UI;
using Assets.Common.Scripts;


/// <summary>
/// Using one-model to train sequentially.
/// </summary>
public class HFReal_PreviewPointRandomTrain : Agent
{
    public GameObject head;
    public Transform target;
    public Transform assistPoints;
    private Camera camera;
    private VehicleController vehicleController;
    private Rigidbody agentRb;
    private Vector3 originalPos;
    private Quaternion originalRot;

    private const int MAX_STEPS = 1024;
    private float MAX_SPEED;
    private float speed;
    private string[] colliderObjectTag = { "Wall", "Person", "Obstacle", "Car" };
    //private string[] fRayObjectTag = { "Wall, Road" };
    private const float RAY_DISTANCE = 15f; //15f
    private const int INFO_DIM = 18;
    //private const int FRAY_DIM = 22;
    private const int RAY_DIM = 64;
    private const int OB_DIM = INFO_DIM  + RAY_DIM;
    private const int AC_DIM = 2;
    private const int RAY_OBSTACLE_START = INFO_DIM+RAY_DIM / 2 - 3;
    private const int RAY_OBSTACLE_END = INFO_DIM + RAY_DIM / 2 + 3;
    private float[] ob;
    private float[] rayAngles;
    private float[] fRayAngles;
    private float[] previousAction;

    ///A-Star
    private const int PATH_NUM = 1;
    private readonly string[] fileName = { "hfreal_path_xz.txt" };
    private FileInfo fw;
    private List<Point>[] pathPoints;
    private string[] pointsString;
    private List<List<float>>[] data;
    private List<Point>[] signList;
    private int[] signNum = new int[PATH_NUM];
    private Preview preview;
    
    Point originalPoint;
    Point targetPoint;
    const float CONST_TARGET_DISTANCE = 8f;
    private List<GameObject> previewPoints;
    Vector3 agentPos;
    private const int PREVIEW_NUMBER = 6;
    private const int PREVIEW_INTERVAL = 15;
    private const float CONST_PREVIEW_DISTANCE = 2.5f;
    public Text txt;
    private float relativeTargetRaw;

    private const int START_NUM = 9;
    private int[] startPosId;
    private Quaternion[] startRot;
    private int r;
    private int startId;
    private int COMPLETE_TARGET_NUM = 15;
    private int completeNum = 0;
    private float COMPLETE_DIS = 3f;
    private float distanceToTarget;
    private float maxDistanceToTarget;
    private bool isFinished = false;
    private bool isCollide = false;
    private int pathId;
    private bool arriveEnd;
    private float randomShift = 1.5f;

    ///for test
    //static FileInfo file;
    //static StreamWriter writer;


    public override void InitializeAgent()
    {
        UnityEngine.Random.InitState(0);
        camera = Camera.main;
        vehicleController = this.GetComponent<VehicleController>();
        agentRb = this.GetComponent<Rigidbody>();
        originalPos = this.transform.position;
        originalRot = this.transform.rotation;
        MAX_SPEED = vehicleController.maxSpeedForward;
        speed = vehicleController.sleepVelocity;
        ob = new float[OB_DIM];
        rayAngles = new float[RAY_DIM];
        //fRayAngles = new float[FRAY_DIM];
        previousAction = new float[AC_DIM];

        pathPoints = new List<Point>[PATH_NUM];
        
        data = new List<List<float>>[PATH_NUM];
        signList = new List<Point>[PATH_NUM];
        for (int i = 0; i < PATH_NUM; i++)
        {
            pathPoints[i] = new List<Point>();
            data[i] = new List<List<float>>();
            signList[i] = new List<Point>();
        }
        preview = new Preview();

        //Debug.Log("assistPoints.childCount: " + assistPoints.childCount);

        ///A star算法计算最短路径
        for (int i=0; i<assistPoints.childCount; i++)
        {
            fw = new FileInfo(fileName[i]);
            if (!fw.Exists)
            {
                Debug.Log("Create path...");
                Vector3 lastStartPos = Vector3.zero;
                int c = 0;
                foreach (Transform child in assistPoints.GetChild(i))
                {
                    if (c >= 20)
                    {
                        AStar aStar = new AStar(lastStartPos, child.transform.position);
                        pathPoints[i].AddRange(aStar.Run());
                        if (c == 24)
                            break;
                    }
                    lastStartPos = child.transform.position;
                    //Debug.Log("lastStartPos: " + lastStartPos.x + ", " + lastStartPos.z);
                    c++;
                }

                StreamWriter sw = fw.CreateText();
                for (int j = 0; j < pathPoints[i].Count; j++)
                {
                    string s = string.Join(" ", pathPoints[i][j].X, " ", pathPoints[i][j].Z);
                    sw.WriteLine(s);
                }
                sw.Close();
                sw.Dispose();
            }
            else
            {
                Debug.Log("From file read path.");
                pointsString = File.ReadAllLines(fileName[i]);
                for (int j = 0; j < pointsString.Length; j++)
                {
                    var x = Regex.Split(pointsString[j].Trim(), "\\s+");
                    pathPoints[i].Add(new Point(Convert.ToSingle(x[0]), Convert.ToSingle(x[1])));

                    //GameObject ga = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                    //ga.transform.position = new Vector3(pathPoints[i][j].X, 0.2f, pathPoints[i][j].Z);
                    //ga.GetComponent<SphereCollider>().enabled = false;
                    //ga.transform.localScale = new Vector3(0.5f, 0.5f, 0.5f);
                }
            }

            data[i] = preview.LoadData(fileName[i]);
            signList[i] = preview.MakePoints(data[i]);
            preview.GetCurvature(signList[i]);
            preview.GetDirectionVector(signList[i]);
            foreach (Point _ in signList[i])
                signNum[i]++;
        }

        startPosId = new int[START_NUM];
        startRot = new Quaternion[START_NUM];
        startPosId[0] = 0;
        startRot[0] = new Quaternion(0.0f, 0.7f, 0.0f, 0.7f);
        startPosId[1] = 1800;
        startRot[1] = new Quaternion(0.0f, 0.5f, 0.0f, 0.866f);
        startPosId[2] = 2550;
        startRot[2] = new Quaternion(0.0f, 0.5f, 0.0f, 0.866f);
        startPosId[3] = 3750;
        startRot[3] = new Quaternion(0.0f, 0.5f, 0.0f, 0.866f);
        startPosId[4] = 4850;
        startRot[4] = new Quaternion(0.0f, 0.609f, 0.0f, 0.793f);
        startPosId[5] = 5750;
        startRot[5] = new Quaternion(0.0f, -0.301f, 0.0f, 0.954f);
        startPosId[6] = 8500;
        startRot[6] = new Quaternion(0.0f, 0.834f, 0.0f, -0.552f);
        startPosId[7] = 10500;
        startRot[7] = new Quaternion(0.0f, 0.834f, 0.0f, -0.552f);
        startPosId[8] = 11400;
        startRot[8] = new Quaternion(0.0f, 1.0f, 0.0f, 0.0f);
        r = UnityEngine.Random.Range(0, START_NUM);
        r = 0;
        startId = startPosId[r] + UnityEngine.Random.Range(0, 30);
        Vector3 pos = new Vector3(signList[pathId][startId].X, 0.01f, signList[pathId][startId].Z);
        pos.x += UnityEngine.Random.Range(-1.5f, 1.5f);
        pos.y = 0.01f;
        pos.z += UnityEngine.Random.Range(-1.5f, 1.5f);
        this.transform.position = pos;
        this.transform.rotation = startRot[r];
        originalPos = this.transform.position;
        originalRot = this.transform.rotation;
        this.agentRb.angularVelocity = Vector3.zero;
        this.agentRb.velocity = Vector3.zero;
        float rangeVelovity = UnityEngine.Random.Range(-5f, -2f);
        //this.agentRb.velocity = (MAX_SPEED + rangeVelovity) * this.transform.forward;
        ///select one path 
        pathId = 0;
        originalPoint = signList[pathId][startId];
        //Debug.Log("originalPoint: (" + originalPoint.X + ", " + originalPoint.Z + ")");
        //previewDistance = preview.GetPreveiwDistance(0, originalPoint, Preview.DEFAULT_PREVIEW_DISTANCE);
        targetPoint = preview.GetPreviewPoint(CONST_TARGET_DISTANCE, originalPoint, signList[pathId]);
        target.transform.position = new Vector3(targetPoint.X, 0.5f, targetPoint.Z);
        maxDistanceToTarget = GetDistanceToTarget();
        distanceToTarget = maxDistanceToTarget;

        int num = 0;
        //for (float i = 30f; i < 150f; i += 360 / 64f)
        //{
        //    fRayAngles[num] = i;
        //    num++;
        //}

        num = 0;
        for (float i = 0f; i < 180f; i += 180f / RAY_DIM)
        {
            rayAngles[num] = i;
            num++;
        }

        //steerModel = new SteerModel();

        previewPoints = new List<GameObject>();
        for (int i = 0; i < PREVIEW_NUMBER; i++)
            previewPoints.Add(GameObject.CreatePrimitive(PrimitiveType.Sphere));

        //ChangeTarget();
    }


    public override void AgentReset()
    {
        //Debug.Log("cr: " + GetCumulativeReward());
    }


    public override void CollectObservations()
    {
        ///forward direction
        relativeTargetRaw = CalculateRelativeRow();
        ob[0] = relativeTargetRaw;

        ///steer
        ob[1] = previousAction[0];

        ///throttle and brake
        ob[2] = previousAction[1];

        ///velocity
        ob[3] = vehicleController.speed;

        //Debug.Log("速度：" + ob[3]);

        ///Add relative position of target
        ob[4] = target.transform.position.x - this.transform.position.x;
        ob[5] = target.transform.position.z - this.transform.position.z;

        ///Add relative position of next 10 preview points
        int start = 6;
        //Debug.Log("targetPoint.Id: " + targetPoint.Id);
        Point lastPreview = targetPoint;
        for (int i = start, j=targetPoint.Id, k = 0; i < start+2*PREVIEW_NUMBER; i+=2, ++k)
        {
            while(preview.PointDistance(lastPreview.X, signList[pathId][j].X,
                lastPreview.Z, signList[pathId][j].Z) < CONST_PREVIEW_DISTANCE)
            {
                j = (j+1) % signList[pathId].Count;
            }
            lastPreview = signList[pathId][j];
            ob[i] = signList[pathId][j].X - this.transform.position.x;
            ob[i + 1] = signList[pathId][j].Z - this.transform.position.z;
            previewPoints[k].transform.position = new Vector3(signList[pathId][j].X, 
                0.2f, 
                signList[pathId][j].Z);

            previewPoints[k].GetComponent<SphereCollider>().enabled = false;
            previewPoints[k].transform.localScale = new Vector3(0.5f, 0.5f, 0.5f);
            Renderer render = previewPoints[k].GetComponent<Renderer>();
            render.material.color = Color.green;
        }

        //string s="";
        for (int i = 4; i < 18; i+=2)
        {
            ob[i] *= -1;
            ob[i + 1] *= -1;
            //s += ob[i] + ", " + ob[i + 1] + ", dis: " + Mathf.Sqrt(Mathf.Pow(ob[i], 2) + Mathf.Pow(ob[i + 1], 2));
        }

        //Debug.Log(s);

        //for (int i = 4; i < 18; i += 2)
        //{
        //    Debug.Log(Mathf.Sqrt(Mathf.Pow(ob[i], 2) + Mathf.Pow(ob[i + 1], 2)));
        //}

        ///Feasible area detection
        //Raycasting(ref ob, this.gameObject, fRayAngles, fRayObjectTag, INFO_DIM, 10f, 0.8f, 106 / 180f * Mathf.PI, Color.blue);

        ///ray-casting
        SphereCasting(ref ob, this.gameObject, rayAngles, INFO_DIM, RAY_DISTANCE, 1f, Color.red);

        ///Add gaussian noise
        //Debug.Log(GetNumberInNormalDistribution(0f, 1f));
        for (int i = INFO_DIM; i < OB_DIM; ++i)
        {
            if (ob[i] > 0)
                ob[i] += Mathf.Clamp(GetNumberInNormalDistribution(0, 1), -1f, 1f);
        }

        ///Add observaations
        AddVectorObs(ob);

        //string obStr = "";
        //for (int i = 0; i < OB_DIM; i++)
        //    obStr += ob[i] + "   ";
        //Debug.Log(obStr);
    }


    public override void AgentAction(float[] vectorAction, string textAction)
    {
        //Debug.Log("rotation: " + transform.rotation.x + ", " + transform.rotation.y + ", " + transform.rotation.z + ", " + transform.rotation.w);
        //Debug.Log("forward: " + transform.forward.x + ", " + transform.forward.y + ", " + transform.forward.z);
        //Debug.Log("forward: " + Vector3.forward.x + ", " + Vector3.forward.y + ", " + Vector3.forward.z);
        //if (GetStepCount() % 5 == 0)
        //    writer.WriteLine(this.transform.position.x + ", " + this.transform.position.z);

        //Debug.Log("target: " + target.transform.position + ",   targetPoint: (" + targetPoint.X + ", " + targetPoint.Z + ")");
        //Debug.Log("Agent pos: " + transform.position);

        float newDistanceToTarget = GetDistanceToTarget();
        string str = "Speed: " + vehicleController.speed +
            "\nTarget Distance: " + newDistanceToTarget +
            "\nSteer: " + vectorAction[0] +
            "\nThrottle And Brake: " + vectorAction[1] +
            "\nRelative Target Raw: " + relativeTargetRaw +
            "\nCumulative Reward: " + GetCumulativeReward();
        txt.text = str;

        if (isCollide)
        {
            AddReward(-12f);
            ChangeTarget();
            isCollide = false;
            Done();
        }
        else if (GetStepCount() > MAX_STEPS)
        {
            ChangeTarget();
            Done();
        }
        else
        {
            ///If car is getting closer to target, the reward is positive, otherwise negtive.
            float progressiveReward = (distanceToTarget - newDistanceToTarget);
            if(newDistanceToTarget < distanceToTarget)
                AddReward(2f * progressiveReward);
            else
                AddReward(3f * progressiveReward);
            distanceToTarget = newDistanceToTarget;

            ///disturbance punishment
            for (int i = 0; i < AC_DIM; i++)
            {
                if(i==0)
                {
                    float disturbance = Mathf.Abs(vectorAction[i]/previousAction[i]-1);
                    if (Mathf.Abs(previousAction[i])>0.1f && disturbance > 0.1f)
                        AddReward(-disturbance/5f);
                    previousAction[i] = vectorAction[i];
                }
                else
                {
                    float disturbance = Mathf.Abs(vectorAction[i]/previousAction[i]-1);
                    if (Mathf.Abs(previousAction[i])>0.1f && disturbance > 0.1f)
                        AddReward(-disturbance/30f);
                    previousAction[i] = vectorAction[i];
                }

            }

            if(vehicleController.speed>10f)
            {
                AddReward(0.1f);
            }
            ///the velocity of car is bigger when closing to human, the punishment is bigger.
            //Debug.Log("startIndex: " + startIndex);
            float minDis = float.MaxValue;
            for (int i = RAY_OBSTACLE_START; i <= RAY_OBSTACLE_END; ++i)
            {
                //Debug.Log(i + "  " + ob[i] + "  " + ob[i + 1] + "  " + ob[i + 2]);
                if (ob[i] > 0 && ob[i] < minDis)
                {
                    minDis = ob[i];
                }
            }
            if (minDis > 0 && minDis < 10f)
            {
                //Debug.Log(minDis);
                float velovityDistanceRate = -vehicleController.speed / minDis;
                AddReward(velovityDistanceRate / 6f);
                //Debug.Log(punish);
            }

            vehicleController.handbrakeInput = 0;
            vehicleController.steerInput = vectorAction[0];
            ///make throttle always max.
            //vehicleController.throttleInput = 1f;
            //vehicleController.brakeInput = 0f;
            if (vectorAction[1] > 0)
            {
                vehicleController.throttleInput = vectorAction[1];
                //vehicleController.throttleInput = 1f;
                vehicleController.brakeInput = 0f;
            }
            else
            {
                vehicleController.throttleInput = 0f;
                vehicleController.brakeInput = -vectorAction[1];
            }

            ///arrive target
            if (distanceToTarget < COMPLETE_DIS)
            {
                if (!isFinished)
                    AddReward(10f);

                isFinished = true;
                ChangeTarget();
                Done();
            }
        }
        //Debug.Log("EularAngle.y: " + transform.eulerAngles.y);
        //Debug.Log("rotation.y: " + transform.rotation.y);
        //Debug.Log("forward.y: " + this.transform.forward.y);
        //Debug.Log("x: " + targetPoint.DirectionVector.x + "   z: " + targetPoint.DirectionVector.z);
    }


    private void OnCollisionEnter(Collision collision)
    {
        for (int i = 0; i < colliderObjectTag.Length; i++)
        {
            if (collision.gameObject.CompareTag(colliderObjectTag[i]))
            {
                isCollide = true;
                break;
            }
        }
    }


    //void OnDrawGizmosSelected()
    //{
    //    for (int i = 0; i < rayAngles.Length; i++)
    //    {
    //        Vector3 origin = this.transform.position;
    //        Vector3 direction = this.transform.TransformDirection(qrray.PolarToCartesian2(20f, rayAngles[i]));
    //        origin.y = 0.8f;
    //        float sphereRadius = 0.5f;
    //        Gizmos.color = Color.red;
    //        //Debug.DrawLine(origin, direction, Color.red);
    //        Gizmos.DrawWireSphere(origin + direction, sphereRadius);
    //    }
    //}


    public void ChangeTarget()
    {
        if (isCollide || GetStepCount() > MAX_STEPS)
        {
            completeNum = 0;
            this.transform.position = originalPos;
            this.transform.rotation = originalRot;
            this.agentRb.angularVelocity = Vector3.zero;
            this.agentRb.velocity = Vector3.zero;
            if (r > 0)
            {
                ///using random start velocity.
                float rangeVelovity = UnityEngine.Random.Range(-5f, -2f);
                this.agentRb.velocity = (MAX_SPEED + rangeVelovity) * this.transform.forward;
                //this.agentRb.velocity = new Vector3(
                //    (MAX_SPEED + rangeVelovity) * Mathf.Sin(this.transform.eulerAngles.y / 180f * Mathf.PI),
                //    0f,
                //    (MAX_SPEED + rangeVelovity) * Mathf.Cos(this.transform.eulerAngles.y / 180f * Mathf.PI));
            }
            //previewDistance = Preview.DEFAULT_PREVIEW_DISTANCE;
            targetPoint = preview.GetPreviewPoint(CONST_TARGET_DISTANCE, originalPoint, signList[pathId]);
        }
        else
        {
            completeNum++;
            if (completeNum == COMPLETE_TARGET_NUM || arriveEnd)
            {
                //Debug.Log("step: " + GetStepCount());
                //Debug.Log("pr: " + (GetCumulativeReward() - 10f));
                r = UnityEngine.Random.Range(0, START_NUM);
                //r = (r + 1) % START_NUM;
                //if (r >= START_NUM)
                //    r = 6;

                //r = 1;
                //Debug.Log("r: " + r);
                startId = startPosId[r] + UnityEngine.Random.Range(0, 30);
                //Debug.Log("startId: " + startId);
                Vector3 pos = new Vector3(signList[pathId][startId].X, 0.01f, signList[pathId][startId].Z);
                pos.x += UnityEngine.Random.Range(-1.5f, 1.5f);
                pos.y = 0.01f;
                pos.z += UnityEngine.Random.Range(-1.5f, 1.5f);
                this.transform.position = pos;
                this.transform.rotation = startRot[r];
                this.agentRb.angularVelocity = Vector3.zero;
                this.agentRb.velocity = Vector3.zero;
                if (r > 0)
                {
                    float rangeVelovity = UnityEngine.Random.Range(-5f, -2f);
                    this.agentRb.velocity = (MAX_SPEED + rangeVelovity) * this.transform.forward;
                    //this.agentRb.velocity = new Vector3(
                    //    (MAX_SPEED + rangeVelovity) * Mathf.Sin(this.transform.eulerAngles.y / 180f * Mathf.PI),
                    //    0f,
                    //    (MAX_SPEED + rangeVelovity) * Mathf.Cos(this.transform.eulerAngles.y / 180f * Mathf.PI));
                }
                originalPos = this.transform.position;
                originalRot = this.transform.rotation;
                originalPoint = signList[pathId][startId];
                //previewDistance = Preview.DEFAULT_PREVIEW_DISTANCE;
                targetPoint = preview.GetPreviewPoint(CONST_TARGET_DISTANCE, originalPoint, signList[pathId]);
                completeNum = 0;
                if (arriveEnd)
                    arriveEnd = false;
            }
            else
            {
                //previewDistance = preview.GetPreveiwDistance(0, targetPoint, previewDistance);
                targetPoint = preview.GetPreviewPoint(CONST_TARGET_DISTANCE, targetPoint, signList[pathId]);
            }
        }

        target.transform.position = new Vector3(targetPoint.X, 0.5f, targetPoint.Z);
        maxDistanceToTarget = GetDistanceToTarget();
        //Debug.Log("dis: " + dis);
        //maxDistanceToTarget = Vector2.Distance(new Vector2(this.transform.position.x, this.transform.position.z),
        //            new Vector2(target.transform.position.x, target.transform.position.z));
        //Debug.Log("maxDistanceToTarget: " + maxDistanceToTarget);
        distanceToTarget = maxDistanceToTarget;

        isFinished = false;
        isCollide = false;
        //Debug.Log("target id: " + targetPoint.Id);
    }


    public void Raycasting(ref float[] ob, GameObject obj, float[] rayAngles, string[] detectObjects,
            int startIndex, float rayDistance, float originHeight, float pitch, Color color)
    {
        Vector3 origin;
        Vector3 direction;
        origin = obj.transform.position;
        RaycastHit hit;
        int index = 0;
        float dis = 0f;
        string ray = "";
        foreach (float angle in rayAngles)
        {
            direction = obj.transform.TransformDirection(new Vector3(rayDistance * Mathf.Sin(pitch) * Mathf.Cos(angle / 180f * Mathf.PI),
                rayDistance * Mathf.Sin(pitch),
                rayDistance * Mathf.Sin(pitch) * Mathf.Sin(angle / 180f * Mathf.PI)));
            //direction.y = endOffset;
            //origin.y = pitch;

            ///drawing ray in the unity editor.
            if (Application.isEditor)
            {
                Debug.DrawRay(origin, direction, color, 0.08f, true);
            }

            if (Physics.Raycast(origin, direction, out hit, rayDistance))
            {
                foreach (string tag in detectObjects)
                {
                    if (hit.collider.gameObject.CompareTag(tag))
                    {
                        dis = hit.distance;
                        break;
                    }
                }

            }

            ray = ray + dis + "  ";
            ob[index + startIndex] = dis;
            index++;
            dis = 0f;
        }

        Debug.Log(ray);
    }


    public void SphereCasting(ref float[] ob, GameObject obj, float[] rayAngles,
            int startIndex, float rayDistance, float endOffset, Color color)
    {
        Vector3 origin = obj.transform.position;
        Vector3 direction;
        RaycastHit hit;
        int index = 0;
        float dis = 0f;
        float sphereRadius = 0.5f;
        string ray = "";

        foreach (float angle in rayAngles)
        {
            direction = obj.transform.TransformDirection(PolarToCartesian2(rayDistance, angle));
            origin.y = endOffset;

            if (Application.isEditor)
            {
                if(startIndex + index >= RAY_OBSTACLE_START && startIndex + index <= RAY_OBSTACLE_END)
                    Debug.DrawRay(origin, direction, Color.blue, 0.08f, true);
                else
                    Debug.DrawRay(origin, direction, Color.red, 0.08f, true);
            }

            if (Physics.SphereCast(origin, sphereRadius, direction, out hit, rayDistance))
                dis = hit.distance;

            ray = ray + dis + "  ";
            ob[startIndex + index] = dis;
            index += 1;
            dis = 0f;
        }
        //Debug.Log(ray);
    }


    public Vector3 PolarToCartesian2(float radius, float angle)
    {
        float x = radius * Mathf.Cos(DegreeToRadian2(angle));
        float z = radius * Mathf.Sin(DegreeToRadian2(angle));
        return new Vector3(x, 0f, z);
    }


    public static float DegreeToRadian2(float degree)
        {
            return degree * Mathf.PI / 180f;
        }


    //private float GetDistanceToTarget()
    //{
    //    float d;
    //    if (Mathf.Abs(targetPoint.DirectionVector.x) > Mathf.Epsilon)
    //    {
    //        float k = -targetPoint.DirectionVector.z / targetPoint.DirectionVector.x;
    //        d = Mathf.Abs(k * this.transform.position.z - this.transform.position.x - k * targetPoint.Z + targetPoint.X) /
    //           Mathf.Sqrt(k * k + 1);

    //        //GameObject ga = GameObject.CreatePrimitive(PrimitiveType.Sphere);
    //        //ga.transform.position = new Vector3(targetPoint.X + 2 / Mathf.Sqrt(k * k + 1) * Mathf.Sin(Mathf.Sqrt(k * k + 1)), 0.2f,
    //        //    targetPoint.Z + 2 / Mathf.Sqrt(k * k + 1) * Mathf.Cos(Mathf.Sqrt(k * k + 1)));
    //        //ga.GetComponent<SphereCollider>().enabled = false;
    //        //ga.transform.localScale = new Vector3(1f, 1f, 1f);
    //        //Debug.Log("d: " + d);
    //    }
    //    else
    //    {
    //        d = Mathf.Abs(this.transform.position.z - targetPoint.Z);
    //        //GameObject ga = GameObject.CreatePrimitive(PrimitiveType.Sphere);
    //        //ga.transform.position = new Vector3(targetPosition.x + 2, 0.2f, targetPosition.z);
    //        //ga.GetComponent<SphereCollider>().enabled = false;
    //        //ga.transform.localScale = new Vector3(0.5f, 0.5f, 0.5f);
    //    }

    //    return d;
    //}


    private float GetDistanceToTarget()
    {
        return Mathf.Sqrt(Mathf.Pow(this.transform.position.x - target.transform.position.x, 2) +
            Mathf.Pow(this.transform.position.z - target.transform.position.z, 2));
    }


    private float CalculateRelativeRow()
    {
        Vector2 target_vec = new Vector2(targetPoint.DirectionVector.z, targetPoint.DirectionVector.x);
        Vector2 current_vec = new Vector2(transform.forward.z, transform.forward.x);
        float dot = Vector2.Dot(current_vec, target_vec);
        float theta;
        if (dot < -1)
            theta = Mathf.PI;
        else if (dot > 1)
            theta = 0f;
        else
            theta = Mathf.Acos(dot);
        float crossProduct = current_vec.x * target_vec.y - current_vec.y * target_vec.x;
        if (crossProduct < 0)
            return -Mathf.Abs(theta);
        else
            return Mathf.Abs(theta);
    }


    private float GetNumberInNormalDistribution(float mean, float std)
    {
        return mean + (RandomNormalDistribution() * std);
    }


    private float RandomNormalDistribution()
    {
        float u = 0.0f, v = 0.0f, w = 0.0f, c = 0.0f;
        do
        {
            u = UnityEngine.Random.Range(-1f, 1f);
            v = UnityEngine.Random.Range(-1f, 1f);
            w = u * u + v * v;
        } while (w == 0.0 || w >= 1.0);
        c = Mathf.Sqrt((-2 * Mathf.Log(w)) / w);
        return u * c;
    }
}
