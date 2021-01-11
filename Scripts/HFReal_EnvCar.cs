using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using EVP;


public class HFReal_EnvCar : MonoBehaviour
{
    private VehicleController vehicleController;
    private Rigidbody agentRb;
    private Vector3 originalPos;
    private Quaternion originalRot;
    public Transform target;
    private const int RAY_DIM = 5;
    private const int OB_DIM = RAY_DIM * 3;
    private float[] ob;
    private float[] rayAngles;
    private string[] colliderObjectTag = { "Wall", "Person", "Obstacle", "Car" };
    private string[] rayObjectTag = { "Wall", "Person", "Car" };
    private const float RAY_DISTANCE = 7f;
    private bool isCollide = false;
    private const int MAX_STEP = 2000;
    private int stepCount = 0;
    private int timeSlice = 0;


    // Start is called before the first frame update
    void Start()
    {
        vehicleController = this.GetComponent<VehicleController>();
        agentRb = this.GetComponent<Rigidbody>();
        originalPos = this.transform.position;
        originalRot = this.transform.rotation;
        ob = new float[OB_DIM];
        rayAngles = new float[RAY_DIM];

        int num = 0;
        for (float i = 80f; i <= 100f; i += 5f)
        {
            rayAngles[num] = i;
            num++;
        }
    }


    // Update is called once per frame
    void Update()
    {
        stepCount++;
        SphereCasting(ref ob, this.gameObject, rayAngles, rayObjectTag, 0, RAY_DISTANCE, 0.8f, Color.red);
        bool tag = false;
        for (int i = 0; i < OB_DIM; i++)
        {
            if (ob[i] > 0)
            {
                tag = true;
                break;
            }
        }
        //Debug.Log("tag: " + tag);
        if (tag || timeSlice > 0)
        {
            --timeSlice;
            vehicleController.throttleInput = 0f;
            vehicleController.brakeInput = 1f;
        }
        else
        {
            vehicleController.throttleInput = 1f;
            vehicleController.brakeInput = 0f;
        }

        float distanceToTarget = Vector2.Distance(new Vector2(transform.position.x, transform.position.z),
            new Vector2(target.position.x, target.position.z));
        if (isCollide || stepCount > MAX_STEP || distanceToTarget < 3f)
        {
            this.transform.position = originalPos;
            this.transform.rotation = originalRot;
            this.agentRb.angularVelocity = Vector3.zero;
            this.agentRb.velocity = Vector3.zero;
            isCollide = false;
            stepCount = 0;
            if (distanceToTarget < 3f)
                timeSlice = 500;
        }
    }


    public void SphereCasting(ref float[] ob, GameObject obj, float[] rayAngles, string[] detectObjects,
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
                Debug.DrawRay(origin, direction, Color.red, 0.08f, true);
            }

            if (Physics.SphereCast(origin, 0.5f, direction, out hit, rayDistance))
                dis = hit.distance;

            //ray = ray + dis + "  " + colliderType[0] + "  " + colliderType[1] + "  ";
            ob[startIndex + index] = dis;
            ++index;
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
}
