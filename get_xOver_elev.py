#!/usr/bin/env python3
# ----------------------------------
# get_xOver_elev.py
#
# Description: Compute crossover position (2 steps procedure: first roughly locate,
# downsampling data to 'msrm_sampl', then with full sampling around the
# points located with the first pass). Finally, retrieve from ladata the interpolated
# elevation
# 
# ----------------------------------
# Author: Stefano Bertone
# Created: 22-Oct-2018

import numpy as np

from intersection import intersection


def get_xOver_elev(arg):
    # Decimate data
    x, y, i, j = intersection(ladata_df.loc[ladata_df['orbID'] == arg[0]][['X_stgprj']].values[::msrm_sampl],
                              ladata_df.loc[ladata_df['orbID'] == arg[0]][['Y_stgprj']].values[::msrm_sampl],
                              ladata_df.loc[ladata_df['orbID'] == arg[1]][['X_stgprj']].values[::msrm_sampl],
                              ladata_df.loc[ladata_df['orbID'] == arg[1]][['Y_stgprj']].values[::msrm_sampl])

    # reassign index to full list (from down-sampled index)
    i *= msrm_sampl
    j *= msrm_sampl

    if (len(x) > 0):

        # save first rough location
        x_raw = x
        y_raw = y

        # compute more accurate location
        x, y, ii, jj = np.squeeze([intersection(ladata_df.loc[ladata_df['orbID'] == arg[0]][['X_stgprj']].values[
                                                max(0, int(i[k]) - msrm_sampl):min(int(i[k]) + msrm_sampl, len(
                                                    ladata_df.loc[ladata_df['orbID'] == arg[0]][['X_stgprj']].values))],
                                                ladata_df.loc[ladata_df['orbID'] == arg[0]][['Y_stgprj']].values[
                                                max(0, int(i[k]) - msrm_sampl):min(int(i[k]) + msrm_sampl, len(
                                                    ladata_df.loc[ladata_df['orbID'] == arg[0]][['Y_stgprj']].values))],
                                                ladata_df.loc[ladata_df['orbID'] == arg[1]][['X_stgprj']].values[
                                                max(0, int(j[k]) - msrm_sampl):min(int(j[k]) + msrm_sampl, len(
                                                    ladata_df.loc[ladata_df['orbID'] == arg[1]][['X_stgprj']].values))],
                                                ladata_df.loc[ladata_df['orbID'] == arg[1]][['Y_stgprj']].values[
                                                max(0, int(j[k]) - msrm_sampl):min(int(j[k]) + msrm_sampl, len(
                                                    ladata_df.loc[ladata_df['orbID'] == arg[1]][['Y_stgprj']].values))])
                                   for k in range(0, len(x))]).T

        # plot and check differences
        if (debug):
            plt.plot(ladata_df.loc[ladata_df['orbID'] == arg[0]][['X_stgprj']].values,
                     ladata_df.loc[ladata_df['orbID'] == arg[0]][['Y_stgprj']].values, c='b')
            plt.plot(ladata_df.loc[ladata_df['orbID'] == arg[1]][['X_stgprj']].values,
                     ladata_df.loc[ladata_df['orbID'] == arg[1]][['Y_stgprj']].values, c='C9')

            plt.plot(ladata_df.loc[ladata_df['orbID'] == arg[0]][['X_stgprj']].values[
                     max(0, int(i[0]) - msrm_sampl):min(int(i[0]) + msrm_sampl, len(
                         ladata_df.loc[ladata_df['orbID'] == arg[0]][['X_stgprj']].values))],
                     ladata_df.loc[ladata_df['orbID'] == arg[0]][['Y_stgprj']].values[
                     max(0, int(i[0]) - msrm_sampl):min(int(i[0]) + msrm_sampl, len(
                         ladata_df.loc[ladata_df['orbID'] == arg[0]][['X_stgprj']].values))], c='r')
            plt.plot(ladata_df.loc[ladata_df['orbID'] == arg[1]][['X_stgprj']].values[
                     max(0, int(j[0]) - msrm_sampl):min(int(j[0]) + msrm_sampl, len(
                         ladata_df.loc[ladata_df['orbID'] == arg[1]][['X_stgprj']].values))],
                     ladata_df.loc[ladata_df['orbID'] == arg[1]][['Y_stgprj']].values[
                     max(0, int(j[0]) - msrm_sampl):min(int(j[0]) + msrm_sampl, len(
                         ladata_df.loc[ladata_df['orbID'] == arg[1]][['Y_stgprj']].values))], c='g')
            plt.plot(x, y, '*k')
            plt.plot(x_raw, y_raw, 'kv')

            delta = 100.
            if (abs(np.amin(np.absolute(x))) > 100.):
                xmin = np.amin(x) - delta
                xmax = np.amax(x) + delta
            else:
                xmax = 200
                xmin = -200
            plt.xlim(xmin, xmax)

            if (abs(np.amin(np.absolute(y))) > 100.):
                ymin = np.amin(y) - delta
                ymax = np.amax(y) + delta
            else:
                ymax = 200
                ymin = -200
            plt.ylim(ymin, ymax)

            plt.savefig('img/intersect_' + arg[0] + '_' + arg[1] + '.png')
            plt.clf()
            plt.close()

        # Retrieve ladata_df index of observations involved in the crossover
        # (can be used to extract orbit number with join btw ladata_df and xovers_df -
        # eg, (ladata_df.loc[ind0][['orbID']].values).reshape(1,-1) -
        # the orbit number can then be used to get the value at i and j by interpolation)
        # ind0 and ind1 now are the indeces of the points just before the
        # intersection in ladata_df, so that (ind0,ind0+1) and (ind1,ind1+1) are the
        # bracketing points' indeces
        if (len(i) > 1):
            ind_A = np.squeeze([ladata_df.loc[ladata_df['orbID'] == arg[0]].iloc[
                                max(0, int(int(np.floor(k))) - msrm_sampl):min(int(int(np.floor(k))) + msrm_sampl, len(
                                    ladata_df.loc[ladata_df['orbID'] == arg[0]][['X_stgprj']].values))].iloc[
                                    int(np.floor(l))][['genID']].values for k, l in zip(i, ii)])
            ind_B = np.squeeze([ladata_df.loc[ladata_df['orbID'] == arg[1]].iloc[
                                max(0, int(int(np.floor(k))) - msrm_sampl):min(int(int(np.floor(k))) + msrm_sampl, len(
                                    ladata_df.loc[ladata_df['orbID'] == arg[1]][['X_stgprj']].values))].iloc[
                                    int(np.floor(l))][['genID']].values for k, l in zip(j, jj)])
        else:
            ind_A = np.squeeze([ladata_df.loc[ladata_df['orbID'] == arg[0]].iloc[
                                max(0, int(int(np.floor(i))) - msrm_sampl):min(int(int(np.floor(i))) + msrm_sampl, len(
                                    ladata_df.loc[ladata_df['orbID'] == arg[0]][['X_stgprj']].values))].iloc[
                                    int(np.floor(ii))][['genID']].values])
            ind_B = np.squeeze([ladata_df.loc[ladata_df['orbID'] == arg[1]].iloc[
                                max(0, int(int(np.floor(j))) - msrm_sampl):min(int(int(np.floor(j))) + msrm_sampl, len(
                                    ladata_df.loc[ladata_df['orbID'] == arg[1]][['X_stgprj']].values))].iloc[
                                    int(np.floor(jj))][['genID']].values])

        # Compute elevation R at crossover points by interpolation
        # (should be put in a function and looped over)

        xyint = np.array([ladata_df.loc[ladata_df['orbID'] == arg[0]].iloc[
                          max(0, int(int(np.floor(k))) - msrm_sampl):min(int(int(np.floor(k)))
                                                                         + msrm_sampl, len(
                              ladata_df.loc[ladata_df['orbID'] == arg[0]][['X_stgprj']].values))][
                              ['genID', 'R']].values.T for k in (np.atleast_1d(i))])

        f_interp = [interpolate.interp1d(xyint[k][0], xyint[k][1], kind='cubic') for k in range(0, len(i))]
        ind_A += np.modf(ii)[0]
        R_A = [f_interp[k](ind_A.item(k)) for k in range(0, len(i))]

        xyint = np.array([ladata_df.loc[ladata_df['orbID'] == arg[1]].iloc[
                          max(0, int(int(np.floor(k))) - msrm_sampl):min(int(int(np.floor(k)))
                                                                         + msrm_sampl, len(
                              ladata_df.loc[ladata_df['orbID'] == arg[1]][['X_stgprj']].values))][
                              ['genID', 'R']].values.T for k in (np.atleast_1d(j))])
        f_interp = [interpolate.interp1d(xyint[k][0], xyint[k][1], kind='cubic') for k in range(0, len(j))]
        ind_B += np.modf(jj)[0]
        R_B = [f_interp[k](ind_B.item(k)) for k in range(0, len(j))]

        if (debug):
            xnew = np.arange(6196, 6215, 0.01)
            ynew = f(xnew)  # use interpolation function returned by `interp1d`
            ynew1 = f1(xnew)  # use interpolation function returned by `interp1d`

            plt.plot(x, y, 'o', xnew, ynew, '-', xnew, ynew1, '--b')
            plt.savefig('test.png')
            plt.clf()
            plt.close()

        # print((x,y,i,ii,j,jj,R_A,R_B))

        return np.vstack((x, y, ind_A, ind_B, R_A, R_B)).T
